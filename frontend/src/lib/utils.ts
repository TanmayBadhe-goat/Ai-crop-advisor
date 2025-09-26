import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";
import { API_CONFIG } from '@/config/api';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

// API configuration
export const API_BASE_URL = API_CONFIG.BASE_URL;

// API helper functions
export const api = {
  async get(endpoint: string) {
    const url = `${API_BASE_URL}${endpoint}`;
    console.log('GET request to:', url);
    
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`API Error: ${response.status} - ${response.statusText}`);
    }
    return response.json();
  },

  async post(endpoint: string, data: any) {
    const url = `${API_BASE_URL}${endpoint}`;
    console.log('POST request to:', url);
    console.log('Request data:', data);
    
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    });
    
    console.log('Response status:', response.status, response.statusText);
    
    if (!response.ok) {
      const errorText = await response.text();
      console.error('Error response:', errorText);
      throw new Error(`API Error: ${response.status} - ${response.statusText}`);
    }
    
    const result = await response.json();
    console.log('Response data:', result);
    return result;
  },

  async postFormData(endpoint: string, formData: FormData) {
    const url = `${API_BASE_URL}${endpoint}`;
    console.log('POST FormData request to:', url);
    
    const response = await fetch(url, {
      method: 'POST',
      body: formData,
    });
    
    console.log('Response status:', response.status, response.statusText);
    
    if (!response.ok) {
      const errorText = await response.text();
      console.error('Error response:', errorText);
      throw new Error(`API Error: ${response.status} - ${response.statusText}`);
    }
    
    const result = await response.json();
    console.log('Response data:', result);
    return result;
  }
};