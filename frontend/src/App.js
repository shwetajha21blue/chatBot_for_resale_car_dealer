import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';
import ReactMarkdown from "react-markdown";
import { v4 as uuidv4 } from 'uuid';

const BACKEND_URL = process.env.REACT_APP_API_URL;
console.log(BACKEND_URL);

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [sessionId, setSessionId] = useState(uuidv4());
  const [loading, setLoading] = useState(false);
  const [sessions, setSessions] = useState([]);
  const [sessionSummaries, setSessionSummaries] = useState({});
  const [showWelcome, setShowWelcome] = useState(true);

  const sendMessage = async () => {
    if (!input.trim()) return;
    setShowWelcome(false);

    const newMessages = [...messages, { sender: 'user', text: input }];
    setMessages(newMessages);
    setLoading(true);
    setInput('');

    if (messages.length === 0) {
      setSessionSummaries((prev) => ({
        ...prev,
        [sessionId]: input.slice(0, 20)
      }));
    }

    // if (messages.length === 0) {
    //   setSessionSummaries(prev => ({
    //     ...prev,
    //     [sessionId]: input.length > 20 
    //       ? `${input.substring(0, 20)}...` 
    //       : input
    //   }));
    // }


    try {
      const response = await axios.post(`${BACKEND_URL}/chat`, {
        question: input,
        session_id: sessionId
      });

      const botReply = response.data.answer;
      setMessages([...newMessages, { sender: 'bot', text: botReply }]);

    } catch (err) {
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  // const handleRefresh = async () => {
  //   try {
  //     const response = await axios.post(`${BACKEND_URL}/refresh`, {
  //       session_id: sessionId
  //     });
  //     console.log(response.data.message);
  //     setMessages([]);
  //     setSessionId(uuidv4());
  //     setSessionSummaries((prev) => ({
  //       ...prev,
  //       [sessionId]: "New Chat "
  //     }));
  //     setShowWelcome(true);
  //   } catch (err) {
  //     if (err.response && err.response.data.detail) {
  //       console.error("Error:", err.response.data.detail);
  //       alert(`Error: ${err.response.data.detail}`);
  //     } else {
  //       console.error("Error:", err.message);
  //       alert("Something went wrong while refreshing the chat.");
  //     }
  //   }
  // };

  const handleDeleteSession = async (id) => {
    try {
      await axios.post(`${BACKEND_URL}/refresh`, { session_id: id });
      setSessions((prev) => prev.filter((s) => s !== id));
      if (id === sessionId) handleNewChat();
    } catch (error) {
      console.error("Error deleting session:", error);
    }
  };

  const fetchMessages = async (id) => {
    try {
      const response = await axios.get(`${BACKEND_URL}/get_messages?session_id=${id}`);
      const data = response.data;
      setMessages(data);
      if (data.length > 0) setShowWelcome(false);
    } catch (error) {
      if (error.response && error.response.status === 404) {
        setMessages([]);
      } else {
        console.error("Load error", error);
      }
    }
  };

  useEffect(() => {
    if (sessionId) {
      fetchMessages(sessionId);
    }
  }, [sessionId]);

  const handleNewChat = async () => {
    try {
      const response = await axios.post(`${BACKEND_URL}/new_chat`);
      const newSessionId = response.data.session_id;
      setSessionId(newSessionId);
      setMessages([]);
      setSessions((prev) => [newSessionId, ...prev]);
      setSessionSummaries((prev) => ({
        ...prev,
        [newSessionId]: "New Chat "
      }));
      setShowWelcome(true);
    } catch (error) {
      console.error("Failed to create new chat:", error);
    }
  };

  const handleSelectSession = (id) => {
    setSessionId(id);
  };

  useEffect(() => {
    const createInitialSession = async () => {
      try {
        const response = await axios.post(`${BACKEND_URL}/new_chat`);
        const newSessionId = response.data.session_id;
        setSessionId(newSessionId);
        setMessages([]);
        setSessions([newSessionId]);
      } catch (error) {
        console.error("Error creating session on load:", error);
      }
    };
    createInitialSession();
  }, []);

  useEffect(() => {
    const chatContainer = document.querySelector('.chat-bot');
    if (chatContainer) {
      chatContainer.scrollTop = chatContainer.scrollHeight;
    }
  }, [messages, showWelcome]);

  return (
    <div className="chat-container">
      <div className="side-bar">
        <div className="sidebar-header">
          <h2>CarGuru AI</h2>
          <p>Your Value Our Agent Assistant</p>
        </div>
        <div className="sidebar-section">
          <h3>Customs Collection</h3>
          <button className='newChat' onClick={handleNewChat}>+ New Chat</button>
        </div>
        <div className="session-list">
          {sessions.map((id) => (
            <div key={id} className={`session-item ${id === sessionId ? "active" : ''}`} onClick={() => handleSelectSession(id)}>
              <div className="session-content">
                {sessionSummaries[id] || "New Chat"}
                <button className='deleteButton' onClick={(e) => {
                  e.stopPropagation();
                  handleDeleteSession(id);
                }}>×</button>
              </div>
            </div>
          ))}
        </div>
        <div className="sidebar-footer">
          <p>All videos enjoy my tablet, help. <strong>See instructions</strong></p>
        </div>
      </div>
      <div className='main'>
        {showWelcome && messages.length === 0 ? (
          <div className="welcome-screen">
            <div className="welcome-header">
              <h1>Welcome to CarGuru AI!</h1>
              <p>Ask me anything about cars, or in the – from images recommendations to obtained comparisons!</p>
            </div>
            
            <div className="action-buttons">
              <button className="action-btn">Get an answer 99 pages</button>
              <button className="action-btn">Write the information</button>
              <button className="action-btn">Write a story twice</button>
              <button className="action-btn">Write an impression online</button>
            </div>
            
            <div className="info-section">
              <button className="learn-more">Learn More for future...</button>
            </div>
            
            <div className="features-grid">
              <div className="feature-card">
                <h3>Three Competitors</h3>
                <p>Get your own people for $1 billion on results.</p>
              </div>
              
              <div className="feature-card">
                <h3>Total Efficiency</h3>
                <p>Choose money and saving costs.</p>
              </div>
              
              <div className="feature-card">
                <h3>57 Insights</h3>
                <p>Excites what we expect for our project.</p>
              </div>
            </div>
          </div>
        ) : (
          <div className='chat-bot'>
            {messages.map((msg, i) => (
              <div key={i} className={`chat-message ${msg.sender === 'bot' ? "bot" : 'user'}`}>
                <div className='message-bubble'>
                  {msg.sender === 'user' ? (
                    <p>{msg.text}</p>
                  ) : (
                    <ReactMarkdown>{msg.text}</ReactMarkdown>
                  )}
                </div>
              </div>
            ))}

            {loading && (
              <div className="loading-indicator">
                <div className="loading-dot"></div>
                <div className="loading-dot"></div>
                <div className="loading-dot"></div>
              </div>
            )}
            
            {/* {messages.length > 0 && (
              <div className='refresh'>
                <button className='refresh-button' onClick={handleRefresh}>Clear Conversation</button>
              </div>
            )} */}
          </div>
        )}
        
        <div className="input-area">
          <input 
            type="text"
            value={input}
            placeholder="Ask something..."
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && sendMessage()}
          />
          <button onClick={sendMessage}>Send</button>
        </div>
      </div>
    </div>
  );
}
