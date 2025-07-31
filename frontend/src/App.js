import React, {useState, useEffect} from 'react'
import axios from 'axios';
import './App.css'
import ReactMarkdown from "react-markdown"
import {v4 as uuidv4} from 'uuid';

const BACKEND_URL = process.env.REACT_APP_API_URL;
console.log(BACKEND_URL)

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [sessionId, setSessionId] = useState(uuidv4());
  const [loading, setLoading] = useState(false);
  const [sessions, setSessions] = useState([])
  const [sessionSummaries, setSessionSummaries] = useState({});
  const [showWelcome, setShowWelcome] = useState(true);

  const sendMessage = async () => {
    if (!input.trim()) return;
    setShowWelcome(false);

    const newMessages = [...messages, {sender:'user', text:input}];
    setMessages(newMessages);
    setLoading(true);
    setInput('');

    if (messages.length === 0){
      setSessionSummaries((prev) => ({
        ...prev,
        [sessionId]: input.slice(0,20)
      }));
    }
    
    try{
      const response = await axios.post(`${BACKEND_URL}/chat`, {
        question: input,
        session_id: sessionId
      });

      const botReply = response.data.answer;
      setMessages([...newMessages, {sender:'bot', text:botReply}]);
      
    } catch (err){
      console.error('Error:', err)
    } finally {
      setLoading(false);
    }
  };

  const handleRefresh = async () => {
    try{
      const response = await axios.post(`${BACKEND_URL}/refresh`,{
        session_id: sessionId
      });
      console.log(response.data.message)
      setMessages([]);
      setSessionId(uuidv4());
      setShowWelcome(true);
    } catch (err) {
      if (err.response && err.response.data.detail) {
        console.error("Error:", err.response.data.detail);
        alert(`Error: ${err.response.data.detail}`);
      } else {
        console.error("Error:", err.message);
        alert("Something went wrong while refreshing the chat.");
      }
    }
  };

  const handleDeleteSession = async (id) => {
    try {
      await axios.post(`${BACKEND_URL}/refresh`, {session_id: id});
      setSessions((prev) => prev.filter((s) => s !== id));
      if(id === sessionId) handleNewChat();
    } catch(error) {
      console.error("Error deleting session:", error)
    }
  };

  const fetchMessages = async (id) => {
    try {
      const response = await axios.get(`${BACKEND_URL}/get_messages?session_id=${id}`);
      const data = response.data
      setMessages(data)
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
      const response = await axios.post(`${BACKEND_URL}/new_chat`)
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
      console.error("Failed to create new chat:", error)
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
        <div>
          <button className='newChat' onClick={handleNewChat}>💬 Chat with AI</button>
        </div>
        <ul className='session'>
          {sessions.map((id) => (
            <li key={id} className='list'>
              <div onClick={() => handleSelectSession(id)}>
                {id === sessionId && <strong> 🔵 </strong> } 
                {sessionSummaries[id] || "New Chat "}
                <button className='deleteButton' onClick={(e) => {
                    e.stopPropagation();
                    handleDeleteSession(id);
                  }}>🗑️</button><br/>
              </div>
            </li>
          ))}
        </ul>
      </div>
      <div className='main'>
        <h2 className='chatBot-heading'>🧠 AutoGuide</h2>
        <div className='chat-bot'>
          {showWelcome && messages.length === 0 && (
            <div className="welcome-screen">
              <h1>Welcome to AutoGuide</h1>
              <p>How can I assist you with your used car needs today?</p>
              
              <div className="welcome-options">
                <div className="welcome-card">
                  <h3>Find Cars by Budget</h3>
                  <p>Discover quality used cars that fit your price range and preferences</p>
                </div>
                
                <div className="welcome-card">
                  <h3>Vehicle History Check</h3>
                  <p>Get detailed reports on accident history, ownership, and service records</p>
                </div>
                
                <div className="welcome-card">
                  <h3>Price Comparison</h3>
                  <p>Compare prices for similar models in your area to get the best deal</p>
                </div>

                <div className="welcome-card">
                  <h3>Inspection Checklist</h3>
                  <p>Learn what to look for when inspecting a used car before purchase</p>
                </div>
              </div>
              
              <div className="upgrade-section">
                <div className="upgrade-card">
                  <h3>Premium Services 🚗💎</h3>
                  <p>Upgrade for personalized car recommendations, detailed market analysis, and expert buying advice.</p>
                  <button className="learn-more">Learn More</button>
                </div>
                <p className="community-text">Join our car enthusiast community for more tips. Join AutoForum</p>
              </div>
            </div>
          )}
          
          {messages.map((msg, i) => (
            <div key={i} className={`chat-message ${msg.sender === 'bot' ? "bot" : 'user'}`}>
              <div className='message-bubble'>
                {msg.sender === 'user' ? (
                  <p>{msg.text}</p>
                ):(
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
          
          {messages.length > 0 && (
            <div className='refresh'>
              <button className='refresh-button' onClick={handleRefresh}>Clear</button>
            </div>
          )}
        </div> 
        
        <div className="input-area">
          <input 
            type="text"
            value={input}
            placeholder="Ask something..."
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key ==="Enter" && sendMessage()}
          />
          <button onClick={sendMessage}>Send</button>
        </div>
      </div>
    </div>
  )
}
