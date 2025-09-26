import axios from 'axios';
import { Platform } from 'react-native';

// Base URL for the backend API
// Use Railway production URL for both development and production
const API_URL = 'https://web-production-af45d.up.railway.app/api';

console.log('Using API URL:', API_URL);
console.log('Platform:', Platform.OS);

// Types
export interface CropPredictionRequest {
  nitrogen: number;
  phosphorus: number;
  potassium: number;
  temperature: number;
  humidity: number;
  ph: number;
  rainfall: number;
}

export interface CropInfo {
  emoji: string;
  season: string;
  duration: string;
  yield: string;
  market_price: string;
  tips: string;
}

export interface CropPredictionResponse {
  success: boolean;
  prediction: {
    crop: string;
    confidence: number;
    emoji: string;
  };
  crop_info: CropInfo;
}

export interface WeatherRequest {
  latitude: number;
  longitude: number;
}

export interface WeatherResponse {
  success: boolean;
  location: {
    city: string;
    country: string;
  };
  current: {
    temperature: number;
    humidity: number;
    condition: string;
    windSpeed: number;
    precipitation: number;
  };
  forecast: Array<{
    date: string;
    maxTemp: number;
    minTemp: number;
    condition: string;
  }>;
  agricultural_advisory: Array<{
    title: string;
    description: string;
  }>;
}

export interface ChatbotRequest {
  message: string;
  lang?: string;
  concise?: boolean;
}

export interface ChatbotResponse {
  success: boolean;
  response: string;
}

export interface DiseaseDetectionResponse {
  success: boolean;
  disease: {
    name: string;
    confidence: number;
    severity: string;
    emoji: string;
  };
  diagnosis: {
    description: string;
    treatment: string;
    prevention: string;
  };
}

export interface DashboardStatsResponse {
  success: boolean;
  stats: {
    total_predictions: {
      value: string;
      growth: string;
    };
    farmers_helped: {
      value: string;
      growth: string;
    };
    crop_varieties: {
      value: string;
      growth: string;
    };
    success_rate: {
      value: string;
      growth: string;
    };
  };
  last_updated: string;
}

// API functions
export const cropService = {
  // Get crop recommendation
  predictCrop: async (data: CropPredictionRequest): Promise<CropPredictionResponse> => {
    try {
      const response = await axios.post(`${API_URL}/predict`, data);
      return response.data;
    } catch (error) {
      console.error('Error predicting crop:', error);
      throw error;
    }
  },

  // Get weather data
  getWeather: async (data: WeatherRequest): Promise<WeatherResponse> => {
    try {
      const response = await axios.post(`${API_URL}/weather`, data);
      return response.data;
    } catch (error) {
      console.error('Error fetching weather:', error);
      throw error;
    }
  },

  // Send message to chatbot
  sendChatMessage: async (data: ChatbotRequest): Promise<ChatbotResponse> => {
    try {
      const response = await axios.post(`${API_URL}/chatbot`, data, {
        headers: {
          'Content-Type': 'application/json',
        },
      });
      return response.data;
    } catch (error) {
      console.error('Error sending chat message:', error);
      throw error;
    }
  },

  // Upload image for disease detection
  detectDisease: async (imageUri: string): Promise<DiseaseDetectionResponse> => {
    try {
      console.log('Starting disease detection for image:', imageUri);
      
      // Create FormData for React Native
      const formData = new FormData();
      
      // For React Native, we need to handle the image differently
      const imageFile = {
        uri: imageUri,
        type: 'image/jpeg',
        name: 'plant_image.jpg',
      };
      
      // Append the image to FormData
      formData.append('image', imageFile as any);

      console.log('Uploading image to:', `${API_URL}/upload-image`);
      
      // First upload the image
      const uploadResponse = await axios.post(`${API_URL}/upload-image`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 30000, // 30 second timeout
      });

      console.log('Image upload response:', uploadResponse.data);

      if (!uploadResponse.data.success || !uploadResponse.data.image_base64) {
        throw new Error('Failed to upload image or get base64 data');
      }

      console.log('Sending to disease detection endpoint');
      
      // Then detect disease using the uploaded image
      const diseaseResponse = await axios.post(`${API_URL}/disease-detection`, {
        image_base64: uploadResponse.data.image_base64,
      }, {
        headers: {
          'Content-Type': 'application/json',
        },
        timeout: 30000, // 30 second timeout
      });

      console.log('Disease detection response:', diseaseResponse.data);

      if (!diseaseResponse.data.success) {
        throw new Error('Disease detection failed');
      }

      return diseaseResponse.data;
    } catch (error) {
      console.error('Error detecting disease:', error);
      
      // Provide more specific error information
      if (axios.isAxiosError(error)) {
        if (error.response) {
          console.error('Response error:', error.response.status, error.response.data);
          throw new Error(`Server error: ${error.response.status} - ${error.response.data?.error || 'Unknown error'}`);
        } else if (error.request) {
          console.error('Request error:', error.request);
          throw new Error('Network error: Unable to reach server');
        } else {
          console.error('Setup error:', error.message);
          throw new Error(`Request setup error: ${error.message}`);
        }
      }
      
      throw error;
    }
  },

  // Get dashboard statistics
  getDashboardStats: async (): Promise<DashboardStatsResponse> => {
    try {
      const response = await axios.get(`${API_URL}/dashboard-stats`, {
        timeout: 8000, // 8 second timeout
      });
      return response.data;
    } catch (error) {
      console.log('Dashboard stats unavailable, using fallback data');
      throw error;
    }
  },
};