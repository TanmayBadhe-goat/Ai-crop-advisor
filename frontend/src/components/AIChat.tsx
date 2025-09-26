import { useState, useRef } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/card";
import { Button } from "@/components/button";
import { Input } from "@/components/input";
import { MessageCircle, Mic, MicOff, Send, Bot, User } from "lucide-react";
import { useToast } from "@/lib/use-toast";
import { api } from "@/lib/utils";

interface ChatMessage {
  id: string;
  type: "user" | "bot";
  message: string;
  timestamp: Date;
}

const AIChat = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: "1",
      type: "bot",
      message: "Hello! I'm your AI farming assistant. Ask me anything about crops, diseases, weather, or farming techniques. You can type or use voice commands!",
      timestamp: new Date(),
    },
  ]);
  const [inputMessage, setInputMessage] = useState("");
  const [isRecording, setIsRecording] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const { toast } = useToast();
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const addMessage = (type: "user" | "bot", message: string) => {
    const newMessage: ChatMessage = {
      id: Date.now().toString(),
      type,
      message,
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, newMessage]);
    setTimeout(scrollToBottom, 100);
  };

  const sendMessage = async () => {
    if (!inputMessage.trim()) return;

    const userMessage = inputMessage;
    setInputMessage("");
    addMessage("user", userMessage);
    setIsLoading(true);

    try {
      // Call your Flask backend chatbot API
      console.log('Sending message to API:', userMessage);
      const response = await api.post('/api/chatbot', {
        message: userMessage,
        lang: 'en-US',
        concise: true
      });

      console.log('API response:', response);

      if (response.success && response.response) {
        addMessage("bot", response.response);
      } else {
        console.error('API returned unsuccessful response:', response);
        addMessage("bot", "Sorry, I couldn't process your request. Please try again.");
      }
    } catch (error) {
      console.error('Chat error details:', error);
      
      // More specific error handling
      if (error instanceof Error) {
        if (error.message.includes('Failed to fetch') || error.message.includes('Network')) {
          addMessage("bot", "I'm having trouble connecting to the server. Please check your internet connection and try again.");
        } else if (error.message.includes('404')) {
          addMessage("bot", "The AI service endpoint was not found. Please contact support.");
        } else if (error.message.includes('500')) {
          addMessage("bot", "The AI service is experiencing issues. Please try again in a few moments.");
        } else {
          addMessage("bot", "I'm having trouble connecting to the AI service. Please try again later.");
        }
      } else {
        addMessage("bot", "An unexpected error occurred. Please try again.");
      }
    } finally {
      setIsLoading(false);
    }
  };

  const toggleRecording = () => {
    if (!isRecording) {
      // Start recording
      if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
        setIsRecording(true);
        toast({
          title: "Recording Started",
          description: "Speak your question now...",
        });
        
        // Simulate recording for demo
        setTimeout(() => {
          setIsRecording(false);
          setInputMessage("What is the best fertilizer for tomatoes?");
          toast({
            title: "Recording Stopped",
            description: "Voice converted to text!",
          });
        }, 3000);
      } else {
        toast({
          title: "Voice Recording Unavailable",
          description: "Speech recognition is not supported in this browser",
          variant: "destructive",
        });
      }
    } else {
      // Stop recording
      setIsRecording(false);
      toast({
        title: "Recording Stopped",
        description: "Processing your voice...",
      });
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <Card className="shadow-card h-96 flex flex-col">
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2">
          <MessageCircle className="h-5 w-5 text-primary" />
          AI Assistant
        </CardTitle>
      </CardHeader>
      <CardContent className="flex-1 flex flex-col p-4">
        <div className="flex-1 overflow-y-auto space-y-3 mb-4">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex gap-2 ${message.type === "user" ? "justify-end" : "justify-start"}`}
            >
              <div
                className={`max-w-[80%] p-3 rounded-lg ${
                  message.type === "user"
                    ? "bg-gradient-primary text-primary-foreground"
                    : "bg-muted text-foreground"
                } animate-fade-in`}
              >
                <div className="flex items-center gap-2 mb-1">
                  {message.type === "user" ? (
                    <User className="h-4 w-4" />
                  ) : (
                    <Bot className="h-4 w-4" />
                  )}
                  <span className="text-xs opacity-80">
                    {message.type === "user" ? "You" : "AI Assistant"}
                  </span>
                </div>
                <p className="text-sm">{message.message}</p>
              </div>
            </div>
          ))}
          {isLoading && (
            <div className="flex justify-start">
              <div className="bg-muted p-3 rounded-lg animate-fade-in">
                <div className="flex items-center gap-2">
                  <Bot className="h-4 w-4" />
                  <div className="flex space-x-1">
                    <div className="w-2 h-2 bg-primary rounded-full animate-bounce"></div>
                    <div className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: "0.1s" }}></div>
                    <div className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: "0.2s" }}></div>
                  </div>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <div className="flex gap-2">
          <div className="flex-1 relative">
            <Input
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask about crops, diseases, weather..."
              className="pr-12"
            />
            <Button
              size="sm"
              variant="ghost"
              className={`absolute right-1 top-1 h-8 w-8 p-0 ${
                isRecording ? "text-destructive animate-pulse-green" : "text-muted-foreground"
              }`}
              onClick={toggleRecording}
            >
              {isRecording ? <MicOff className="h-4 w-4" /> : <Mic className="h-4 w-4" />}
            </Button>
          </div>
          <Button 
            onClick={sendMessage} 
            disabled={!inputMessage.trim() || isLoading}
            className="px-4"
          >
            <Send className="h-4 w-4" />
          </Button>
        </div>
      </CardContent>
    </Card>
  );
};

export default AIChat;