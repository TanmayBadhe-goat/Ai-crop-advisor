import axios from 'axios';
import { Platform } from 'react-native';

// Network diagnostic utility for troubleshooting connectivity issues
export class NetworkDiagnostic {
  private static instance: NetworkDiagnostic;
  
  public static getInstance(): NetworkDiagnostic {
    if (!NetworkDiagnostic.instance) {
      NetworkDiagnostic.instance = new NetworkDiagnostic();
    }
    return NetworkDiagnostic.instance;
  }

  // Test basic internet connectivity with multiple fallbacks
  async testInternetConnectivity(): Promise<{
    success: boolean;
    provider?: string;
    responseTime: number;
  }> {
    const testUrls = [
      'https://www.google.com',
      'https://www.cloudflare.com',
      'https://httpbin.org/get',
      'https://jsonplaceholder.typicode.com/posts/1'
    ];

    for (const url of testUrls) {
      const startTime = Date.now();
      try {
        const response = await axios.get(url, {
          timeout: 8000,
          headers: {
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache'
          }
        });
        const responseTime = Date.now() - startTime;
        if (response.status === 200) {
          console.log(`Internet connectivity test passed with ${url}`);
          return {
            success: true,
            provider: url,
            responseTime
          };
        }
      } catch (error) {
        console.log(`Internet connectivity test failed for ${url}:`, error);
        continue;
      }
    }
    
    return {
      success: false,
      responseTime: 0
    };
  }

  // Get network type information
  private getNetworkType(): string {
    return Platform.OS === 'android' ? 'mobile/wifi' : 'unknown';
  }

  // Test specific API endpoint with detailed error reporting
  async testApiEndpoint(url: string): Promise<{
    success: boolean;
    responseTime: number;
    error?: string;
    statusCode?: number;
    networkType?: string;
  }> {
    const startTime = Date.now();
    try {
      console.log(`Testing API endpoint: ${url}`);
      const response = await axios.get(`${url}/dashboard-stats`, {
        timeout: 15000,
        headers: {
          'Cache-Control': 'no-cache',
          'Pragma': 'no-cache',
          'User-Agent': `KrishiMitra-Mobile/${Platform.OS}`
        }
      });
      const responseTime = Date.now() - startTime;
      console.log(`API test successful: ${response.status} in ${responseTime}ms`);
      return {
        success: response.status === 200,
        responseTime,
        statusCode: response.status,
        networkType: this.getNetworkType()
      };
    } catch (error: any) {
      const responseTime = Date.now() - startTime;
      let errorMessage = 'Unknown error';
      let statusCode: number | undefined;

      if (axios.isAxiosError(error)) {
        if (error.response) {
          errorMessage = `Server error: ${error.response.status} - ${error.response.statusText}`;
          statusCode = error.response.status;
        } else if (error.request) {
          errorMessage = 'Network error: No response received from server';
        } else {
          errorMessage = `Request setup error: ${error.message}`;
        }
      } else {
        errorMessage = error.message || 'Unknown error';
      }

      console.log(`API test failed: ${errorMessage} (${responseTime}ms)`);
      return {
        success: false,
        responseTime,
        error: errorMessage,
        statusCode,
        networkType: this.getNetworkType()
      };
    }
  }

  // Test multiple endpoints for redundancy
  async testMultipleEndpoints(): Promise<{
    railway: { success: boolean; responseTime: number; error?: string };
    fallbackTest: { success: boolean; responseTime: number; error?: string };
  }> {
    const railwayTest = await this.testApiEndpoint('https://web-production-af45d.up.railway.app/api');
    
    // Test a simple HTTP endpoint as fallback
    const fallbackTest = await this.testApiEndpoint('https://httpbin.org');
    
    return {
      railway: railwayTest,
      fallbackTest: fallbackTest
    };
  }

  // Comprehensive network diagnostic with mobile data specific checks
  async runDiagnostic(): Promise<{
    internetConnectivity: { success: boolean; provider?: string; responseTime: number };
    railwayApi: { success: boolean; responseTime: number; error?: string; statusCode?: number };
    fallbackTest: { success: boolean; responseTime: number; error?: string };
    platform: string;
    timestamp: string;
    recommendations: string[];
  }> {
    console.log('Running comprehensive network diagnostic...');
    
    const internetConnectivity = await this.testInternetConnectivity();
    const endpointTests = await this.testMultipleEndpoints();
    
    const recommendations: string[] = [];
    
    // Generate recommendations based on test results
    if (!internetConnectivity.success) {
      recommendations.push('No internet connectivity detected. Check your mobile data or WiFi connection.');
    } else if (!endpointTests.railway.success) {
      if (endpointTests.fallbackTest.success) {
        recommendations.push('Internet works but Railway API is unreachable. This might be a server issue.');
      } else {
        recommendations.push('Limited internet connectivity. Try moving to an area with better signal.');
      }
    }
    
    if (internetConnectivity.responseTime > 5000) {
      recommendations.push('Slow internet connection detected. Consider switching to WiFi if available.');
    }
    
    if (Platform.OS === 'android') {
      recommendations.push('For Android: Ensure mobile data is enabled and app has network permissions.');
    }
    
    const result = {
      internetConnectivity,
      railwayApi: endpointTests.railway,
      fallbackTest: endpointTests.fallbackTest,
      platform: Platform.OS,
      timestamp: new Date().toISOString(),
      recommendations
    };
    
    console.log('Network diagnostic results:', JSON.stringify(result, null, 2));
    return result;
  }

  // Quick connectivity check for app startup
  async quickConnectivityCheck(): Promise<boolean> {
    try {
      const response = await axios.get('https://web-production-af45d.up.railway.app/api/dashboard-stats', {
        timeout: 5000,
      });
      return response.status === 200;
    } catch (error) {
      console.log('Quick connectivity check failed, trying fallback...');
      try {
        const fallbackResponse = await axios.get('https://httpbin.org/get', {
          timeout: 3000,
        });
        return fallbackResponse.status === 200;
      } catch (fallbackError) {
        console.log('All connectivity checks failed');
        return false;
      }
    }
  }
}
