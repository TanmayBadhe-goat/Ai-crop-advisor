// API Configuration
export const API_CONFIG = {
  BASE_URL: import.meta.env.VITE_API_URL || 
           (import.meta.env.PROD ? 'https://web-production-d6596.up.railway.app' : 'http://localhost:5000'),
  ENDPOINTS: {
    PREDICT: '/api/predict',
    CHATBOT: '/api/chatbot',
    WEATHER: '/api/weather',
    UPLOAD_IMAGE: '/api/upload-image',
    DISEASE_DETECTION: '/api/disease-detection',
    TEST_KEYS: '/api/test-keys'
  }
};
