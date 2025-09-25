import axios, { AxiosError, AxiosRequestConfig } from 'axios';
import { Platform } from 'react-native';
import NetInfo from '@react-native-community/netinfo';

// Base URL for the backend API
// Use Railway production URL for both development and production
const RAILWAY_API_URL = 'https://web-production-d6596.up.railway.app/api';
const LOCAL_API_URL = 'http://10.0.2.2:5000/api'; // Android emulator localhost
const LOCALHOST_API_URL = 'http://localhost:5000/api'; // iOS simulator
const PC_LOCAL_URL = 'http://192.168.29.153:5000/api'; // PC local network IP

// Dynamic API URL - will be determined at runtime
let API_URL = RAILWAY_API_URL;
let currentApiUrl = RAILWAY_API_URL;

console.log('Using API URL:', API_URL);
console.log('Platform:', Platform.OS);

// Configure axios defaults for mobile data optimization
axios.defaults.timeout = 20000; // Increased timeout for mobile hotspot connections
axios.defaults.headers.common['User-Agent'] = `KrishiMitra-Mobile/${Platform.OS}`;
axios.defaults.headers.common['Cache-Control'] = 'no-cache';
axios.defaults.headers.common['Pragma'] = 'no-cache';
axios.defaults.headers.common['Connection'] = 'keep-alive';
// Additional headers for mobile hotspot compatibility
axios.defaults.headers.common['Accept'] = 'application/json, text/plain, */*';
axios.defaults.headers.common['Accept-Encoding'] = 'gzip, deflate, br';

// Add request interceptor for better mobile data handling
axios.interceptors.request.use(
  (config) => {
    // Add timestamp to prevent caching issues on mobile data
    if (config.url && !config.url.includes('?')) {
      config.url += `?_t=${Date.now()}`;
    } else if (config.url) {
      config.url += `&_t=${Date.now()}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Add response interceptor for better error handling
axios.interceptors.response.use(
  (response) => response,
  (error) => {
    console.log('Axios interceptor caught error:', error.message);
    return Promise.reject(error);
  }
);

// Enhanced network connectivity test function
const testConnectivity = async (url: string): Promise<boolean> => {
  try {
    // First check device network connectivity
    const netInfo = await NetInfo.fetch();
    if (!netInfo.isConnected) {
      console.log('Device is not connected to internet');
      return false;
    }

    const response = await axios.get(`${url}/dashboard-stats`, {
      timeout: 8000, // Increased timeout for mobile hotspot connectivity
      headers: {
        'Cache-Control': 'no-cache',
        'Pragma': 'no-cache'
      }
    });
    return response.status === 200;
  } catch (error) {
    console.log(`Connectivity test failed for ${url}:`, error);
    return false;
  }
};

// Find the best available API URL
const findBestApiUrl = async (): Promise<string> => {
  console.log('Finding best API URL...');
  
  // Test URLs in order of preference
  const urlsToTest = [
    RAILWAY_API_URL,
    PC_LOCAL_URL,
    Platform.OS === 'android' ? LOCAL_API_URL : LOCALHOST_API_URL
  ];
  
  for (const url of urlsToTest) {
    console.log(`Testing ${url}...`);
    if (await testConnectivity(url)) {
      console.log(`✅ Found working API: ${url}`);
      currentApiUrl = url;
      API_URL = url;
      return url;
    }
  }
  
  console.log('❌ No working API found, using Railway as fallback');
  currentApiUrl = RAILWAY_API_URL;
  API_URL = RAILWAY_API_URL;
  return RAILWAY_API_URL;
};

// Network retry utility with exponential backoff
const retryWithBackoff = async <T>(
  operation: () => Promise<T>,
  maxRetries: number = 3,
  baseDelay: number = 1000
): Promise<T> => {
  let lastError: any;
  
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      // Check network connectivity before each attempt
      const netInfo = await NetInfo.fetch();
      if (!netInfo.isConnected) {
        throw new Error('No internet connection available');
      }
      
      return await operation();
    } catch (error) {
      lastError = error;
      console.log(`Attempt ${attempt + 1}/${maxRetries + 1} failed:`, error);
      
      if (attempt < maxRetries) {
        const delay = baseDelay * Math.pow(2, attempt); // Exponential backoff
        console.log(`Waiting ${delay}ms before retry...`);
        await new Promise(resolve => setTimeout(resolve, delay));
      }
    }
  }
  
  throw lastError;
};

// Enhanced error message generator
const getNetworkErrorMessage = (error: any): string => {
  if (axios.isAxiosError(error)) {
    if (error.code === 'NETWORK_ERROR' || error.message.includes('Network Error')) {
      return 'Network connection failed. Please check your internet connection and try again.';
    }
    if (error.code === 'ECONNABORTED' || error.message.includes('timeout')) {
      return 'Request timed out. Please check your connection and try again.';
    }
    if (error.response?.status === 500) {
      return 'Server error occurred. Please try again later.';
    }
    if (error.response?.status === 404) {
      return 'Service not found. Please update the app or try again later.';
    }
  }
  return error.message || 'An unexpected error occurred. Please try again.';
};

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

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
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

// Get the current API URL (with automatic discovery)
export const getBestApiUrl = async (): Promise<string> => {
  return await findBestApiUrl();
};

// Get current API URL without testing (for display purposes)
export const getCurrentApiUrl = (): string => {
  return currentApiUrl;
};

// API functions
export const cropService = {
  // Get crop recommendation with retry mechanism
  predictCrop: async (data: CropPredictionRequest): Promise<CropPredictionResponse> => {
    const apiUrl = await findBestApiUrl();
    return retryWithBackoff(async () => {
      const response = await axios.post(`${apiUrl}/predict`, data, {
        timeout: 18000, // Increased timeout for mobile hotspot connections
        headers: {
          'Content-Type': 'application/json',
        },
      });
      return response.data;
    }, 2, 1500); // 2 retries with 1.5s base delay
  },

  // Get weather data with retry mechanism
  getWeather: async (data: WeatherRequest): Promise<WeatherResponse> => {
    const apiUrl = await findBestApiUrl();
    return retryWithBackoff(async () => {
      const response = await axios.post(`${apiUrl}/weather`, data, {
        timeout: 18000, // Increased timeout for mobile hotspot connections
        headers: {
          'Content-Type': 'application/json',
        },
      });
      return response.data;
    }, 2, 1500); // 2 retries with 1.5s base delay
  },

  // Send message to chatbot with retry mechanism
  sendChatMessage: async (data: ChatbotRequest): Promise<ChatbotResponse> => {
    const apiUrl = await findBestApiUrl();
    return retryWithBackoff(async () => {
      const response = await axios.post(`${apiUrl}/chatbot`, data, {
        timeout: 15000, // 15 second timeout for chat
        headers: {
          'Content-Type': 'application/json',
        },
      });
      return response.data;
    }, 2, 1000); // 2 retries with 1s base delay
  },

  // Upload image for disease detection with fallback mechanism
  detectDisease: async (imageUri: string): Promise<DiseaseDetectionResponse> => {
    const tryDetectWithUrl = async (apiUrl: string): Promise<DiseaseDetectionResponse> => {
      console.log('Starting disease detection for image:', imageUri);
      console.log('Using API URL:', apiUrl);
      
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

      console.log('Uploading image to:', `${apiUrl}/upload-image`);
      
      // First upload the image
      const uploadResponse = await axios.post(`${apiUrl}/upload-image`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 20000, // Increased timeout for mobile hotspot connections
      });

      console.log('Image upload response:', uploadResponse.data);

      if (!uploadResponse.data.success || !uploadResponse.data.image_base64) {
        throw new Error('Failed to upload image or get base64 data');
      }

      console.log('Sending to disease detection endpoint');
      
      // Then detect disease using the uploaded image
      const diseaseResponse = await axios.post(`${apiUrl}/disease-detection`, {
        image_base64: uploadResponse.data.image_base64,
      }, {
        headers: {
          'Content-Type': 'application/json',
        },
        timeout: 20000, // Increased timeout for mobile hotspot connections
      });

      console.log('Disease detection response:', diseaseResponse.data);

      if (!diseaseResponse.data.success) {
        throw new Error('Disease detection failed');
      }

      return diseaseResponse.data;
    };

    // Retry mechanism with exponential backoff
    const maxRetries = 2;
    let lastError: any;

    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      try {
        console.log(`Attempt ${attempt + 1}/${maxRetries + 1} - Trying Railway API`);
        return await tryDetectWithUrl(RAILWAY_API_URL);
      } catch (error) {
        lastError = error;
        console.warn(`Railway API attempt ${attempt + 1} failed:`, error);
        
        if (attempt < maxRetries) {
          // Wait before retry (exponential backoff)
          const delay = Math.pow(2, attempt) * 1000; // 1s, 2s, 4s...
          console.log(`Waiting ${delay}ms before retry...`);
          await new Promise(resolve => setTimeout(resolve, delay));
        }
      }
    }

    // Try local API as final fallback (for development)
    try {
      console.log('Trying local API as fallback...');
      const localUrl = Platform.OS === 'android' ? LOCAL_API_URL : LOCALHOST_API_URL;
      return await tryDetectWithUrl(localUrl);
    } catch (localError) {
      console.error('Local API also failed:', localError);
      
      // Provide more specific error information based on the last Railway error
      if (axios.isAxiosError(lastError)) {
        if (lastError.response) {
          console.error('Response error:', lastError.response.status, lastError.response.data);
          throw new Error(`Server error: ${lastError.response.status} - ${lastError.response.data?.error || 'Unknown error'}`);
        } else if (lastError.request) {
          console.error('Request error:', lastError.request);
          throw new Error('Network error: Unable to reach server. Please check your internet connection and try again.');
        } else {
          console.error('Setup error:', lastError.message);
          throw new Error(`Request setup error: ${lastError.message}`);
        }
      }
      
      throw new Error('Unable to connect to any server after multiple attempts. Please check your internet connection and try again.');
    }
  },

  // Get dashboard statistics with retry mechanism
  getDashboardStats: async (): Promise<DashboardStatsResponse> => {
    const apiUrl = await findBestApiUrl();
    return retryWithBackoff(async () => {
      const response = await axios.get(`${apiUrl}/dashboard-stats`, {
        timeout: 8000, // 8 second timeout
      });
      return response.data;
    }, 1, 1000); // 1 retry with 1s delay for dashboard stats
  },

  // Check network connectivity
  checkNetworkConnectivity: async (): Promise<boolean> => {
    try {
      const netInfo = await NetInfo.fetch();
      return netInfo.isConnected ?? false;
    } catch (error) {
      console.log('Error checking network connectivity:', error);
      return false;
    }
  },

  // Get network error message
  getNetworkErrorMessage,

  // Test API connectivity
  testApiConnectivity: async (): Promise<boolean> => {
    const apiUrl = await findBestApiUrl();
    return testConnectivity(apiUrl);
  },

  // Get current API URL info
  getApiInfo: () => {
    return {
      currentUrl: currentApiUrl,
      railwayUrl: RAILWAY_API_URL,
      localUrl: Platform.OS === 'android' ? LOCAL_API_URL : LOCALHOST_API_URL,
      pcLocalUrl: PC_LOCAL_URL
    };
  },
};