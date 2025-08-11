import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';
import ReactMarkdown from "react-markdown";
import { v4 as uuidv4 } from 'uuid';


import Box from '@mui/material/Box';
import InputLabel from '@mui/material/InputLabel';
import MenuItem from '@mui/material/MenuItem';
import FormControl from '@mui/material/FormControl';
import Select from '@mui/material/Select';

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
  const [colour, setColour] = React.useState('');
  const [price, setPrice] = React.useState('');
  const [mileage, setMileage] = React.useState('');
  const [gear, setGear] = React.useState('');
  const [reverseCamera, setReverseCamera] = React.useState('');
  const [clearConversation, setClearConversation] = useState(false)
  // const [newChat, setNewChat] = useState(true)

  const sendMessage = async () => {
    if (!input.trim() && !colour && !price && !mileage && !gear && !reverseCamera) return;
    setShowWelcome(false);
    setClearConversation(false);
    // setNewChat(true)

    let messageToSend = input.trim();

    // Append dropdown values to message if present
    if (!messageToSend) {
      messageToSend = [colour, price, mileage, gear, reverseCamera]
      .join(". ");
        // .filter(Boolean)
    }

    const newMessages = [...messages, { sender: 'user', text: messageToSend  }];
    setMessages(newMessages);
    setInput('');
    setColour('');
    setPrice('');
    setMileage('');
    setGear('');
    setReverseCamera('');
    setLoading(true);

    if (messages.length === 0) {
      setSessionSummaries((prev) => ({
        ...prev,
        [sessionId]: messageToSend.slice(0, 20)
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
        question: messageToSend,
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


  const clearChatScreen = () => {
    setMessages([]);
    setClearConversation(true);
  }

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
      // setNewChat(false)
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



  const handleChangeColour = (event) => {
    setColour(event.target.value);

  };
  console.log(colour)

  const handleChangePrice = (event) => {
    setPrice(event.target.value);
  };
  console.log(price)

  const handleChangeMileage = (event) => {
    setMileage(event.target.value);
  };
  console.log(mileage)

  const handleChangeGear = (event) => {
    setGear(event.target.value);
  };
  console.log(gear)

  const handleChangeReverseCamera = (event) => {
    setReverseCamera(event.target.value);
  };
  console.log(reverseCamera)

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
                <div className='sidebar-button'>
                  <button title='Permanently Delete' className='deleteButton' onClick={(e) => {
                    e.stopPropagation();
                    handleDeleteSession(id);
                  }}>🗑️</button>
                  <a
                    href={`http://localhost:8000/export_pdf?session_id=${id}`}
                    title="Download Chat PDF"
                    className="downloadButton"
                    onClick={(e) => e.stopPropagation()} // prevent triggering session change
                    download
                  >
                    ⬇️
                  </a>
                </div>
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
            {/* <div className="welcome-header">
              <h1>CarGuru AI!</h1>
              <p>Ask me anything about cars, or in the – from images recommendations to obtained comparisons!</p>
            </div> */}
            <div className="welcome-input-area">
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
        ) : (
          <div className='chat-bot'>
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
            {messages.map((msg, i) => (
              <>
              <div key={i} className={`chat-message ${msg.sender === 'bot' ? "bot" : 'user'}`}>
                <div className='message-bubble'>
                  {msg.sender === 'user' ? (
                    <p>{msg.text}</p>
                  ) : (
                    <>
                    <ReactMarkdown>{msg.text}
                    </ReactMarkdown>
                    </>
                  )}
                </div>
              </div>
            </>
            ))}
            {(!loading && !clearConversation) && (<div className="dropdown-buttons-container">
              <p>Would you like to further filter your answers by the following categories? </p>
            <div className='categories-options-dropdown'>
              {/* <div className='first-three-categories'> */}
            <Box sx={{ minWidth: 170 }}>
              <FormControl fullWidth>
                <InputLabel id="demo-simple-select-label">Colour </InputLabel>
                <Select
                  labelId="demo-simple-select-label"
                  id="demo-simple-select"
                  value={colour}
                  label="Colour"
                  onChange={handleChangeColour}
                >
                  <MenuItem value={"I want Red color cars from my previous search"}>Red</MenuItem>
                  <MenuItem value={"I want Blue color cars from my previous search"}>Blue</MenuItem>
                  <MenuItem value={"I want White color cars from my previous search"}>White</MenuItem>
                  <MenuItem value={"I want Black color cars from my previous search"}>Black</MenuItem>
                  <MenuItem value={"I want Silver color cars from my previous search"}>Silver</MenuItem>
                </Select>
              </FormControl>
            </Box>

            <Box sx={{ minWidth: 170 }}>
              <FormControl fullWidth>
                <InputLabel id="demo-simple-select-label">Price Range </InputLabel>
                <Select
                  labelId="demo-simple-select-label"
                  id="demo-simple-select"
                  value={price}
                  label="Price Range"
                  onChange={handleChangePrice}
                >
                  <MenuItem value={"I want cars under 10 lakhs from my previous search"}>Under 10 Lakhs</MenuItem>
                  <MenuItem value={"I want cars between 10 to 30 lakhs from my previous search"}>10-30 Lakhs</MenuItem>
                  <MenuItem value={"I want cars between 30 to 50 lakhs from my previous search"}>30-50 Lakhs</MenuItem>
                  <MenuItem value={"I want cars between 50 to 70 lakhs from my previous search"}>50-70 Lakhs</MenuItem>
                  <MenuItem value={"I want cars between 70 to 90 lakhs from my previous search"}>70-90 Lakhs</MenuItem>
                  <MenuItem value={"I want cars above 90 lakhs from my previous search"}>Above 90 Lakhs</MenuItem>
                </Select>
              </FormControl>
            </Box>

            <Box sx={{ minWidth: 170 }}>
              <FormControl fullWidth>
                <InputLabel id="demo-simple-select-label">Mileage </InputLabel>
                <Select
                  labelId="demo-simple-select-label"
                  id="demo-simple-select"
                  value={mileage}
                  label="Mileage"
                  onChange={handleChangeMileage}
                >
                  <MenuItem value={"I want cars under 10 kmpl mileage from my previous search"}>Under 10 kmpl</MenuItem>
                  <MenuItem value={"I want cars between 10 to 20 kmpl mileage from my previous search"}>10-20 kmpl</MenuItem>
                  <MenuItem value={"I want cars between 20 to 30 kmpl mileage from my previous search"}>20-30 kmpl</MenuItem>
                  <MenuItem value={"I want cars between 30 to 40 kmpl mileage from my previous search"}>30-40 kmpl</MenuItem>
                  <MenuItem value={"I want cars above 40 kmpl mileage from my previous search"}>Above 40 kmpl</MenuItem>
                </Select>
              </FormControl>
            </Box>
            {/* </div>
            <div className='last-categories'> */}
            <Box sx={{ minWidth: 170 }}>
              <FormControl fullWidth>
                <InputLabel id="demo-simple-select-label">Gear Type </InputLabel>
                <Select
                  labelId="demo-simple-select-label"
                  id="demo-simple-select"
                  value={gear}
                  label="Gear Type"
                  onChange={handleChangeGear}
                >
                  <MenuItem value={"I want cars which have gear type automatic from my previous search"}>Automatic</MenuItem>
                  <MenuItem value={"I want cars which have gear type manual from my previous search"}>Manual</MenuItem>
                </Select>
              </FormControl>
            </Box>

            
            <Box sx={{ minWidth: 170 }}>
              <FormControl fullWidth>
                <InputLabel id="demo-simple-select-label">Reverse Camera </InputLabel>
                <Select
                  labelId="demo-simple-select-label"
                  id="demo-simple-select"
                  value={reverseCamera}
                  label="Reverse Camera"
                  onChange={handleChangeReverseCamera}
                >
                  <MenuItem value={"I want cars which have Reverse Camera from my previous search"}>Yes</MenuItem>
                  <MenuItem value={"I want cars which don't have Reverse Camera from my previous search"}>No</MenuItem>
                </Select>
              </FormControl>
            </Box>
            {/* </div> */}
            </div>
            <div className='categories-submit-btn'>
              <button 
              type="submit" 
              className="submit-btn"
              onClick={sendMessage}
              >
                ➤
              </button>
            </div>
            </div>)}
            {loading && (
              <div className="loading-indicator">
                <div className="loading-dot"></div>
                <div className="loading-dot"></div>
                <div className="loading-dot"></div>
              </div>
            )}
            {(messages.length > 0 && !loading )&& (
              <div className='refresh'>
                <button 
                className='refresh-button' 
                title='Do you want to clean the chat window.' 
                onClick={clearChatScreen}
                >
                  Clear Conversation
                </button>
              </div>
            )}
          </div>
        )}
        
        {/* <div className="input-area">
          <input 
            type="text"
            value={input}
            placeholder="Ask something..."
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && sendMessage()}
          />
          <button onClick={sendMessage}>Send</button>
        </div> */}
      </div>
    </div>
  );
}
