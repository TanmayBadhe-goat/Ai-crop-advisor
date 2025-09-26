import React, { useState } from 'react';
import { View, StyleSheet, ScrollView, KeyboardAvoidingView, Platform } from 'react-native';
import { TextInput, Button, Card, Title, Text, Avatar } from 'react-native-paper';
import { cropService, ChatMessage } from '../api/cropService';

const FarmAssistantScreen = () => {
  const [message, setMessage] = useState('');
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([
    {
      role: 'assistant',
      content: 'Hello! I\'m your farming assistant. How can I help you today?'
    }
  ]);
  const [loading, setLoading] = useState(false);

  const sendMessage = async () => {
    if (!message.trim()) return;
    
    const userMessage: ChatMessage = {
      role: 'user',
      content: message
    };
    
    setChatHistory(prev => [...prev, userMessage]);
    setMessage('');
    setLoading(true);
    
    try {
      const response = await cropService.sendChatMessage({
        message: message,
        lang: 'en-US',
        concise: true
      });
      
      if (response.success) {
        setChatHistory(prev => [...prev, {
          role: 'assistant',
          content: response.response
        }]);
      } else {
        setChatHistory(prev => [
          ...prev, 
          {
            role: 'assistant',
            content: 'Sorry, I encountered an error. Please try again later.'
          }
        ]);
      }
    } catch (error) {
      console.error(error);
      setChatHistory(prev => [
        ...prev, 
        {
          role: 'assistant',
          content: 'Sorry, I encountered an error. Please try again later.'
        }
      ]);
    } finally {
      setLoading(false);
    }
  };

  const getMessageEmoji = (role: string) => {
    return role === 'user' ? '👨‍🌾' : '🤖';
  };

  return (
    <KeyboardAvoidingView 
      style={styles.container}
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}
      keyboardVerticalOffset={80}
    >
      <View style={styles.header}>
        <Title style={styles.title}>Farm Assistant</Title>
        <Text style={styles.subtitle}>Ask me anything about farming!</Text>
      </View>
      
      <ScrollView 
        style={styles.chatContainer}
        contentContainerStyle={styles.chatContent}
      >
        {chatHistory.map((chat, index) => (
          <View 
            key={index} 
            style={[
              styles.messageBubble,
              chat.role === 'user' ? styles.userBubble : styles.assistantBubble
            ]}
          >
            <View style={styles.messageHeader}>
              <Text style={styles.emoji}>{getMessageEmoji(chat.role)}</Text>
              <Text style={styles.role}>
                {chat.role === 'user' ? 'You' : 'Assistant'}
              </Text>
            </View>
            <Text style={styles.messageText}>{chat.content}</Text>
          </View>
        ))}
      </ScrollView>
      
      <Card style={styles.inputContainer}>
        <Card.Content style={styles.inputContent}>
          <TextInput
            style={styles.input}
            value={message}
            onChangeText={setMessage}
            placeholder="Ask about crops, diseases, or farming tips..."
            multiline
            mode="outlined"
            outlineColor="#4CAF50"
            activeOutlineColor="#2E7D32"
          />
          <Button
            mode="contained"
            onPress={sendMessage}
            loading={loading}
            disabled={loading || !message.trim()}
            style={styles.sendButton}
            icon="send"
          >
            Send
          </Button>
        </Card.Content>
      </Card>
    </KeyboardAvoidingView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F5F5F5',
  },
  header: {
    padding: 16,
    backgroundColor: '#4CAF50',
    alignItems: 'center',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: 'white',
  },
  subtitle: {
    fontSize: 16,
    color: 'rgba(255, 255, 255, 0.8)',
  },
  chatContainer: {
    flex: 1,
  },
  chatContent: {
    padding: 16,
  },
  messageBubble: {
    padding: 12,
    borderRadius: 12,
    marginBottom: 12,
    maxWidth: '85%',
    elevation: 1,
  },
  userBubble: {
    backgroundColor: '#E8F5E9',
    alignSelf: 'flex-end',
    borderBottomRightRadius: 4,
  },
  assistantBubble: {
    backgroundColor: 'white',
    alignSelf: 'flex-start',
    borderBottomLeftRadius: 4,
  },
  messageHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 6,
  },
  emoji: {
    fontSize: 16,
    marginRight: 6,
  },
  role: {
    fontSize: 12,
    fontWeight: 'bold',
    color: '#666',
  },
  messageText: {
    fontSize: 16,
    color: '#333',
  },
  inputContainer: {
    margin: 8,
    elevation: 4,
  },
  inputContent: {
    flexDirection: 'row',
    alignItems: 'flex-end',
  },
  input: {
    flex: 1,
    marginRight: 8,
    backgroundColor: 'white',
  },
  sendButton: {
    backgroundColor: '#4CAF50',
  },
});

export default FarmAssistantScreen;