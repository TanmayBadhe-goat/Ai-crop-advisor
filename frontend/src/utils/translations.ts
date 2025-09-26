// Multi-language support for AI Crop Advisor
export interface Translation {
  // App
  app: {
    title: string;
    tagline: string;
  };
  
  // Navigation
  navigation: {
    dashboard: string;
    weather: string;
    crops: string;
    detection: string;
    chat: string;
    calendar: string;
    analytics: string;
    business: string;
    community: string;
    reports: string;
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
    changeLanguage: string;
    available: string;
    comingSoon: string;
    underDevelopment: string;
    inDevelopment: string;
  };
  
  // Dashboard
  dashboard: {
    hero: {
      title: string;
      subtitle: string;
    };
    stats: {
      farmers: string;
      diseases: string;
      recommendations: string;
      queries: string;
      fromLastMonth: string;
    };
    features: {
      title: string;
      cropRecommendations: {
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
      marketplace: {
        title: string;
        description: string;
      };
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
    season: string;
    duration: string;
    expectedYield: string;
    marketPrice: string;
    farmingTips: string;
  };
  
  // Weather
  weather: {
    title: string;
    windSpeed: string;
    refresh: string;
    humidity: string;
    feelsLike: string;
    pressure: string;
    visibility: string;
  };
  
  // AI Chat
  chat: {
    title: string;
    placeholder: string;
    send: string;
  };
  
  // Disease Detection
  diseaseDetection: {
    title: string;
    uploadPhoto: string;
    chooseImage: string;
    analyzing: string;
    result: string;
    accuracy: string;
    treatment: string;
    prevention: string;
  };
  
  // Analytics
  analytics: {
    title: string;
    totalPredictions: string;
    farmersHelped: string;
    successRate: string;
    cropVarieties: string;
    regionalPerformance: string;
    cropAccuracy: string;
    impactMetrics: string;
    predictionAccuracy: string;
    farmerSatisfaction: string;
    yieldImprovement: string;
    costReduction: string;
  };
  
  // Business Model
  business: {
    title: string;
    mission: string;
    vision: string;
    revenueStreams: string;
    currentImpact: string;
    scalabilityRoadmap: string;
    competitiveAdvantages: string;
  };
}

export const translations: Record<string, Translation> = {
  en: {
    // App
    app: {
      title: "CropAI Assistant",
      tagline: "Smart farming solutions"
    },
    
    // Navigation
    navigation: {
      dashboard: "Dashboard",
      weather: "Weather",
      crops: "Crops",
      detection: "Disease Detection",
      chat: "AI Assistant",
      calendar: "Crop Calendar",
      analytics: "Analytics",
      business: "Business Model",
      community: "Community",
      reports: "Reports",
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
      changeLanguage: "Change language",
      available: "Available",
      comingSoon: "Coming Soon",
      underDevelopment: "This section is under development and will be available soon.",
      inDevelopment: "In Development"
    },
    
    // Dashboard
    dashboard: {
      hero: {
        title: "AI Crop Advisor",
        subtitle: "Empowering farmers with AI-driven insights for better crop management and higher yields"
      },
      stats: {
        farmers: "Active Farmers",
        diseases: "Diseases Detected",
        recommendations: "Crop Recommendations",
        queries: "AI Queries Answered",
        fromLastMonth: "from last month"
      },
      features: {
        title: "Platform Features",
        cropRecommendations: {
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
          description: "Speak to get farming advice in your local language"
        },
        offlineMode: {
          title: "Offline Mode",
          description: "Basic predictions and advice without internet connectivity"
        },
        marketplace: {
          title: "Marketplace Integration",
          description: "Direct access to sell your produce at best prices"
        }
      }
    },
    
    // Crop Prediction
    cropPrediction: {
      title: "Crop Recommendation",
      description: "Enter soil and climate data to get personalized crop recommendations",
      nitrogen: "Nitrogen (N) - kg/hectare",
      phosphorus: "Phosphorus (P) - kg/hectare",
      potassium: "Potassium (K) - kg/hectare",
      temperature: "Temperature (°C)",
      humidity: "Humidity (%)",
      phValue: "pH Value",
      rainfall: "Rainfall (mm)",
      getRecommendation: "Get Recommendation",
      recommendedCrop: "Recommended Crop",
      confidence: "Confidence",
      season: "Season",
      duration: "Duration",
      expectedYield: "Expected Yield",
      marketPrice: "Market Price",
      farmingTips: "Farming Tips"
    },
    
    // Weather
    weather: {
      title: "Live Weather",
      windSpeed: "Wind Speed",
      refresh: "Refresh",
      humidity: "Humidity",
      feelsLike: "Feels Like",
      pressure: "Pressure",
      visibility: "Visibility"
    },
    
    // AI Chat
    chat: {
      title: "AI Farming Assistant",
      placeholder: "Type your message...",
      send: "Send"
    },
    
    // Disease Detection
    diseaseDetection: {
      title: "Disease Detection",
      uploadPhoto: "Upload Plant Photo",
      chooseImage: "Choose Image",
      analyzing: "Analyzing...",
      result: "Detection Result",
      accuracy: "Accuracy",
      treatment: "Recommended Treatment",
      prevention: "Prevention Tips"
    },
    
    // Analytics
    analytics: {
      title: "Analytics & Performance",
      totalPredictions: "Total Predictions",
      farmersHelped: "Farmers Helped",
      successRate: "Success Rate",
      cropVarieties: "Crop Varieties",
      regionalPerformance: "Regional Performance",
      cropAccuracy: "Crop Prediction Accuracy",
      impactMetrics: "Impact Metrics",
      predictionAccuracy: "Prediction Accuracy",
      farmerSatisfaction: "Farmer Satisfaction",
      yieldImprovement: "Yield Improvement",
      costReduction: "Cost Reduction"
    },
    
    // Business Model
    business: {
      title: "Business Model & Scalability",
      mission: "Mission",
      vision: "Vision",
      revenueStreams: "Revenue Streams",
      currentImpact: "Current Impact & Traction",
      scalabilityRoadmap: "3-Year Scalability Roadmap",
      competitiveAdvantages: "Competitive Advantages"
    }
  },
  hi: {
    // App
    app: {
      title: "क्रॉपएआई सहायक",
      tagline: "स्मार्ट खेती समाधान"
    },
    
    // Navigation
    navigation: {
      dashboard: "डैशबोर्ड",
      weather: "मौसम",
      crops: "फसलें",
      detection: "रोग पहचान",
      chat: "एआई सहायक",
      calendar: "फसल कैलेंडर",
      analytics: "विश्लेषण",
      business: "व्यवसाय मॉडल",
      community: "समुदाय",
      reports: "रिपोर्ट्स",
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
      changeLanguage: "भाषा बदलें",
      available: "उपलब्ध",
      comingSoon: "जल्द ही आ रहा है",
      underDevelopment: "यह सेक्शन विकासाधीन है और जल्द ही उपलब्ध होगा।",
      inDevelopment: "विकासाधीन"
    },
    
    // Dashboard
    dashboard: {
      hero: {
        title: "एआई क्रॉप सलाहकार",
        subtitle: "बेहतर फसल प्रबंधन और अधिक उपज के लिए एआई-संचालित अंतर्दृष्टि के साथ किसानों को सशक्त बनाना"
      },
      stats: {
        farmers: "सक्रिय किसान",
        diseases: "रोगों का पता चला",
        recommendations: "फसल सिफारिशें",
        queries: "एआई प्रश्नों के उत्तर",
        fromLastMonth: "पिछले महीने से"
      },
      features: {
        title: "प्लेटफ़ॉर्म सुविधाएँ",
        cropRecommendations: {
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
        marketplace: {
          title: "मार्केटप्लेस एकीकरण",
          description: "अपनी उपज को सर्वोत्तम कीमतों पर बेचने के लिए सीधी पहुंच"
        }
      }
    },
    
    // Crop Prediction
    cropPrediction: {
      title: "फसल सिफारिश",
      description: "व्यक्तिगत फसल सिफारिशों के लिए मिट्टी और जलवायु डेटा दर्ज करें",
      nitrogen: "नाइट्रोजन (N) - किग्रा/हेक्टेयर",
      phosphorus: "फॉस्फोरस (P) - किग्रा/हेक्टेयर",
      potassium: "पोटैशियम (K) - किग्रा/हेक्टेयर",
      temperature: "तापमान (°C)",
      humidity: "नमी (%)",
      phValue: "पीएच मान",
      rainfall: "वर्षा (मिमी)",
      getRecommendation: "सिफारिश प्राप्त करें",
      recommendedCrop: "सुझाई गई फसल",
      confidence: "विश्वसनीयता",
      season: "मौसम",
      duration: "अवधि",
      expectedYield: "अनुमानित उपज",
      marketPrice: "बाजार मूल्य",
      farmingTips: "कृषि युक्तियाँ"
    },
    
    // Weather
    weather: {
      title: "लाइव मौसम",
      windSpeed: "हवा की गति",
      refresh: "ताज़ा करें",
      humidity: "नमी",
      feelsLike: "महसूस हो रहा है",
      pressure: "दबाव",
      visibility: "दृश्यता"
    },
    
    // AI Chat
    chat: {
      title: "एआई कृषि सहायक",
      placeholder: "अपना संदेश टाइप करें...",
      send: "भेजें"
    },
    
    // Disease Detection
    diseaseDetection: {
      title: "रोग पहचान",
      uploadPhoto: "पौधे की फोटो अपलोड करें",
      chooseImage: "छवि चुनें",
      analyzing: "विश्लेषण किया जा रहा है...",
      result: "परिणाम",
      accuracy: "सटीकता",
      treatment: "सुझाया गया उपचार",
      prevention: "रोकथाम के उपाय"
    },
    
    // Analytics
    analytics: {
      title: "विश्लेषण और प्रदर्शन",
      totalPredictions: "कुल भविष्यवाणियाँ",
      farmersHelped: "किसानों की सहायता",
      successRate: "सफलता दर",
      cropVarieties: "फसल किस्में",
      regionalPerformance: "क्षेत्रीय प्रदर्शन",
      cropAccuracy: "फसल भविष्यवाणी सटीकता",
      impactMetrics: "प्रभाव मेट्रिक्स",
      predictionAccuracy: "भविष्यवाणी सटीकता",
      farmerSatisfaction: "किसान संतुष्टि",
      yieldImprovement: "उपज में वृद्धि",
      costReduction: "लागत में कमी"
    },
    
    // Business Model
    business: {
      title: "व्यवसाय मॉडल और विस्तार योजना",
      mission: "हमारा मिशन",
      vision: "हमारी दृष्टि",
      revenueStreams: "आय के स्रोत",
      currentImpact: "वर्तमान प्रभाव",
      scalabilityRoadmap: "3-वर्षीय विस्तार योजना",
      competitiveAdvantages: "प्रतिस्पर्धात्मक लाभ"
    }
  }
};
