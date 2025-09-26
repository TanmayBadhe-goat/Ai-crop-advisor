// Multi-language support for AI Crop Advisor Mobile App
export interface Translation {
  // App
  app: {
    title: string;
    tagline: string;
    welcome: string;
  };
  
  // Navigation
  navigation: {
    dashboard: string;
    weather: string;
    cropPrediction: string;
    diseaseDetection: string;
    aiAssistant: string;
    cropCalendar: string;
    community: string;
    settings: string;
  };
  
  // Common
  common: {
    loading: string;
    error: string;
    success: string;
    submit: string;
    cancel: string;
    reset: string;
    refresh: string;
    available: string;
    comingSoon: string;
    getStarted: string;
    scanPlant: string;
    checkWeather: string;
    language: string;
  };
  
  // Dashboard
  dashboard: {
    welcomeTitle: string;
    subtitle: string;
    lastUpdated: string;
    stats: {
      totalPredictions: string;
      farmersHelped: string;
      cropVarieties: string;
      successRate: string;
    };
    features: {
      title: string;
      smartCropRecommendations: {
        title: string;
        description: string;
      };
      diseaseDetection: {
        title: string;
        description: string;
      };
      weatherIntegration: {
        title: string;
        description: string;
      };
      voiceAssistant: {
        title: string;
        description: string;
      };
      offlineMode: {
        title: string;
        description: string;
      };
      expertConnect: {
        title: string;
        description: string;
      };
    };
    quickStart: {
      title: string;
      steps: string[];
    };
  };
  
  // Crop Prediction
  cropPrediction: {
    title: string;
    description: string;
    nitrogen: string;
    phosphorus: string;
    potassium: string;
    temperature: string;
    humidity: string;
    phValue: string;
    rainfall: string;
    getRecommendation: string;
    recommendedCrop: string;
    confidence: string;
  };
  
  // Weather
  weather: {
    title: string;
    currentWeather: string;
    temperature: string;
    humidity: string;
    windSpeed: string;
    pressure: string;
    visibility: string;
    feelsLike: string;
  };
  
  // Disease Detection
  diseaseDetection: {
    title: string;
    uploadPhoto: string;
    takePhoto: string;
    chooseFromGallery: string;
    analyzing: string;
    result: string;
    accuracy: string;
    treatment: string;
    prevention: string;
  };
  
  // AI Assistant
  aiAssistant: {
    title: string;
    placeholder: string;
    send: string;
    listening: string;
    speakNow: string;
  };
  
  // Settings
  settings: {
    title: string;
    language: string;
    notifications: string;
    about: string;
    help: string;
    logout: string;
  };
}

export const translations: Record<string, Translation> = {
  en: {
    // App
    app: {
      title: "CropAI Assistant",
      tagline: "Smart farming solutions",
      welcome: "Welcome to CropAI Assistant"
    },
    
    // Navigation
    navigation: {
      dashboard: "Dashboard",
      weather: "Weather",
      cropPrediction: "Crop Prediction",
      diseaseDetection: "Disease Detection",
      aiAssistant: "AI Assistant",
      cropCalendar: "Crop Calendar",
      community: "Community",
      settings: "Settings"
    },
    
    // Common
    common: {
      loading: "Loading...",
      error: "Error",
      success: "Success",
      submit: "Submit",
      cancel: "Cancel",
      reset: "Reset",
      refresh: "Refresh",
      available: "Available",
      comingSoon: "Coming Soon",
      getStarted: "Get Started",
      scanPlant: "Scan Plant",
      checkWeather: "Check Weather",
      language: "Language"
    },
    
    // Dashboard
    dashboard: {
      welcomeTitle: "Welcome to CropAI Assistant",
      subtitle: "Empowering farmers with AI-driven agricultural solutions",
      lastUpdated: "Last updated",
      stats: {
        totalPredictions: "Total Predictions",
        farmersHelped: "Farmers Helped",
        cropVarieties: "Crop Varieties",
        successRate: "Success Rate"
      },
      features: {
        title: "Platform Features",
        smartCropRecommendations: {
          title: "Smart Crop Recommendations",
          description: "AI-powered suggestions based on soil, weather, and market conditions"
        },
        diseaseDetection: {
          title: "Disease Detection",
          description: "Upload plant images for instant disease identification and treatment"
        },
        weatherIntegration: {
          title: "Weather Integration",
          description: "Real-time weather data for optimal farming decisions"
        },
        voiceAssistant: {
          title: "Voice Assistant",
          description: "Speak and get farming advice in your local language"
        },
        offlineMode: {
          title: "Offline Mode",
          description: "Basic predictions and advice without internet connectivity"
        },
        expertConnect: {
          title: "Expert Connect",
          description: "Direct access to agricultural experts for complex queries"
        }
      },
      quickStart: {
        title: "Quick Start Guide",
        steps: [
          "Check current weather conditions for your area",
          "Browse crop database for planting recommendations",
          "Upload plant photos for disease detection",
          "Chat with AI assistant for personalized advice"
        ]
      }
    },
    
    // Crop Prediction
    cropPrediction: {
      title: "Crop Recommendation",
      description: "Get personalized crop suggestions based on soil and climate data",
      nitrogen: "Nitrogen (N)",
      phosphorus: "Phosphorus (P)",
      potassium: "Potassium (K)",
      temperature: "Temperature (°C)",
      humidity: "Humidity (%)",
      phValue: "pH Value",
      rainfall: "Rainfall (mm)",
      getRecommendation: "Get Recommendation",
      recommendedCrop: "Recommended Crop",
      confidence: "Confidence"
    },
    
    // Weather
    weather: {
      title: "Weather Information",
      currentWeather: "Current Weather",
      temperature: "Temperature",
      humidity: "Humidity",
      windSpeed: "Wind Speed",
      pressure: "Pressure",
      visibility: "Visibility",
      feelsLike: "Feels Like"
    },
    
    // Disease Detection
    diseaseDetection: {
      title: "Disease Detection",
      uploadPhoto: "Upload Plant Photo",
      takePhoto: "Take Photo",
      chooseFromGallery: "Choose from Gallery",
      analyzing: "Analyzing...",
      result: "Detection Result",
      accuracy: "Accuracy",
      treatment: "Recommended Treatment",
      prevention: "Prevention Tips"
    },
    
    // AI Assistant
    aiAssistant: {
      title: "AI Farming Assistant",
      placeholder: "Type your message...",
      send: "Send",
      listening: "Listening...",
      speakNow: "Speak now"
    },
    
    // Settings
    settings: {
      title: "Settings",
      language: "Language",
      notifications: "Notifications",
      about: "About",
      help: "Help & Support",
      logout: "Logout"
    }
  },
  
  hi: {
    // App
    app: {
      title: "क्रॉपएआई सहायक",
      tagline: "स्मार्ट खेती समाधान",
      welcome: "क्रॉपएआई सहायक में आपका स्वागत है"
    },
    
    // Navigation
    navigation: {
      dashboard: "डैशबोर्ड",
      weather: "मौसम",
      cropPrediction: "फसल भविष्यवाणी",
      diseaseDetection: "रोग पहचान",
      aiAssistant: "एआई सहायक",
      cropCalendar: "फसल कैलेंडर",
      community: "समुदाय",
      settings: "सेटिंग्स"
    },
    
    // Common
    common: {
      loading: "लोड हो रहा है...",
      error: "त्रुटि",
      success: "सफल",
      submit: "जमा करें",
      cancel: "रद्द करें",
      reset: "रीसेट",
      refresh: "ताज़ा करें",
      available: "उपलब्ध",
      comingSoon: "जल्द ही आ रहा है",
      getStarted: "शुरू करें",
      scanPlant: "पौधे को स्कैन करें",
      checkWeather: "मौसम जांचें",
      language: "भाषा"
    },
    
    // Dashboard
    dashboard: {
      welcomeTitle: "क्रॉपएआई सहायक में आपका स्वागत है",
      subtitle: "एआई-संचालित कृषि समाधानों के साथ किसानों को सशक्त बनाना",
      lastUpdated: "अंतिम बार अपडेट किया गया",
      stats: {
        totalPredictions: "कुल भविष्यवाणियाँ",
        farmersHelped: "किसानों की सहायता",
        cropVarieties: "फसल किस्में",
        successRate: "सफलता दर"
      },
      features: {
        title: "प्लेटफ़ॉर्म सुविधाएँ",
        smartCropRecommendations: {
          title: "स्मार्ट फसल सिफारिशें",
          description: "मिट्टी, मौसम और बाजार की स्थिति के आधार पर एआई-संचालित सुझाव"
        },
        diseaseDetection: {
          title: "रोग का पता लगाना",
          description: "तत्काल रोग पहचान और उपचार के लिए पौधे की छवियां अपलोड करें"
        },
        weatherIntegration: {
          title: "मौसम एकीकरण",
          description: "इष्टतम कृषि निर्णयों के लिए रीयल-टाइम मौसम डेटा"
        },
        voiceAssistant: {
          title: "वॉइस असिस्टेंट",
          description: "अपनी स्थानीय भाषा में कृषि सलाह के लिए बोलें"
        },
        offlineMode: {
          title: "ऑफलाइन मोड",
          description: "इंटरनेट कनेक्शन के बिना बुनियादी भविष्यवाणियां और सलाह"
        },
        expertConnect: {
          title: "विशेषज्ञ कनेक्ट",
          description: "जटिल प्रश्नों के लिए कृषि विशेषज्ञों तक सीधी पहुंच"
        }
      },
      quickStart: {
        title: "त्वरित प्रारंभ गाइड",
        steps: [
          "अपने क्षेत्र के लिए वर्तमान मौसम की स्थिति जांचें",
          "रोपण सिफारिशों के लिए फसल डेटाबेस ब्राउज़ करें",
          "रोग का पता लगाने के लिए पौधे की तस्वीरें अपलोड करें",
          "व्यक्तिगत सलाह के लिए एआई सहायक से चैट करें"
        ]
      }
    },
    
    // Crop Prediction
    cropPrediction: {
      title: "फसल सिफारिश",
      description: "मिट्टी और जलवायु डेटा के आधार पर व्यक्तिगत फसल सुझाव प्राप्त करें",
      nitrogen: "नाइट्रोजन (N)",
      phosphorus: "फॉस्फोरस (P)",
      potassium: "पोटैशियम (K)",
      temperature: "तापमान (°C)",
      humidity: "नमी (%)",
      phValue: "पीएच मान",
      rainfall: "वर्षा (मिमी)",
      getRecommendation: "सिफारिश प्राप्त करें",
      recommendedCrop: "सुझाई गई फसल",
      confidence: "विश्वसनीयता"
    },
    
    // Weather
    weather: {
      title: "मौसम जानकारी",
      currentWeather: "वर्तमान मौसम",
      temperature: "तापमान",
      humidity: "नमी",
      windSpeed: "हवा की गति",
      pressure: "दबाव",
      visibility: "दृश्यता",
      feelsLike: "महसूस हो रहा है"
    },
    
    // Disease Detection
    diseaseDetection: {
      title: "रोग पहचान",
      uploadPhoto: "पौधे की फोटो अपलोड करें",
      takePhoto: "फोटो लें",
      chooseFromGallery: "गैलरी से चुनें",
      analyzing: "विश्लेषण किया जा रहा है...",
      result: "परिणाम",
      accuracy: "सटीकता",
      treatment: "सुझाया गया उपचार",
      prevention: "रोकथाम के उपाय"
    },
    
    // AI Assistant
    aiAssistant: {
      title: "एआई कृषि सहायक",
      placeholder: "अपना संदेश टाइप करें...",
      send: "भेजें",
      listening: "सुन रहा है...",
      speakNow: "अब बोलें"
    },
    
    // Settings
    settings: {
      title: "सेटिंग्स",
      language: "भाषा",
      notifications: "सूचनाएं",
      about: "के बारे में",
      help: "सहायता और समर्थन",
      logout: "लॉग आउट"
    }
  }
};

export default translations;
