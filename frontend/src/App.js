import React, {useState, useEffect} from 'react'
// import React, {useState} from 'react'
import axios from 'axios';
import './App.css'
import ReactMarkdown from "react-markdown"
import {v4 as uuidv4} from 'uuid';


export default function App() {
  const [messages, setMessages] = useState([]);  // Store Messages
  const [input, setInput] = useState('');  // Question
  const [sessionId, setSessionId] = useState(uuidv4());   // Session Id
  const [loading, setLoading] = useState(false);   // Loading variable
  const [sessions, setSessions] = useState([])   // Store Chat Histories
  const [sessionSummaries, setSessionSummaries] = useState({});


  const sendMessage = async () => {
    if (!input.trim()) return;

    const newMessages = [...messages, {sender:'user', text:input}];
    setMessages(newMessages);
    setLoading(true); // Show loading
    setInput('');
    
    try{
      const response = await axios.post('http://127.0.0.1:8000/chat', {
        question: input,
        session_id: sessionId
      });

      const botReply = response.data.answer;

      setMessages([...newMessages, {sender:'bot', text:botReply}]);
      setInput('');
      
    } catch (err){
      console.error('Error:', err)
    } finally {
      setLoading(false); // Hide Loading
    }

  };

  
  const handleRefresh = async () => {
    try{
      const response = await axios.post("http://127.0.0.1:8000/refresh",{
        session_id: sessionId
      });
      console.log(response.data.message)
      setMessages([]);
      setSessionId(uuidv4());
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
      await axios.post("http://127.0.0.1:8000/refresh", {session_id: id});
      setSessions((prev) => prev.filter((s) => s !== id));

      if(id === sessionId) handleNewChat();
    } catch(error) {
      console.error("Error deleting session:", error)
    }
  };


  const fetchMessages = async (id) => {
    try {
      const response = await axios.get(`http://127.0.0.1:8000/get_messages?session_id=${id}`);
      const data = response.data
      setMessages(data)
      if (data.length > 0) {
        setSessionSummaries((prev) => ({
          ...prev,
          [id]: data[0].text.slice(0, 20)
        }));
      } else {
        setSessionSummaries((prev) => ({
          ...prev,
          [id]: "New Chat"
        }));
      }
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
      const response = await axios.post("http://127.0.0.1:8000/new_chat")
      const newSessionId = response.data.session_id;
      setSessionId(newSessionId);
      setMessages([]);
      setSessions((prev) => [newSessionId, ...prev]);
      setSessionSummaries((prev) => ({
        ...prev,
        [newSessionId]: "New Chat"
      }));
    } catch (error) {
      console.error("Failed to create new chat:", error)
    }
  };


  const handleSelectSession = (id) => {
    setSessionId(id);
  };


useEffect(() => {
  let hasInitialized = false;

  const createInitialSession = async () => {
    if (hasInitialized) return;
    hasInitialized = true;

    try {
      const response = await axios.post("http://127.0.0.1:8000/new_chat");
      const newSessionId = response.data.session_id;
      setSessionId(newSessionId);
      setMessages([]);
      setSessions([newSessionId]);  // No duplicate
    } catch (error) {
      console.error("Error creating session on load:", error);
    }
  };

  createInitialSession();
}, []);


  return (
    <>
    <div className="chat-container">
      <div className="side-bar">
        <div>
        {/* <strong className='sideHeading' >Chat History</strong> */}
          <button className='newChat' onClick={handleNewChat}>💬 New chat</button>
        </div>
        <ul className='session'>
          {sessions.map((id) => (
            <li key={id} className='list'>
              {/* {sessionSummaries[id] || "New Chat"}
              {id === sessionId && <strong> (active)</strong>}  */}
              <div onClick={() => handleSelectSession(id)}>
                {/* {id} {id === sessionId && <strong>(active)</strong>} */}
                  {/* {messages.length > 0 && id === sessionId ? messages[0].text.slice(0, 20) : id}/ */}
                  {sessionSummaries[id] || "New Chat "}
                  {id === sessionId && <strong> (Active) </strong> } 
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
      <h2>🧠 ChatBot</h2>
      {messages.length > 0 && (
        <>
        <div className='chat-bot'>
        {messages.map((msg, i) => (
          <div key={i}
          className={`chat-message ${msg.sender === 'bot' ? "bot" : 'user'}`}>
            {msg.sender === 'user' ? (
              <p>{msg.text}</p>  // Human Message
              ):(
                <ReactMarkdown>{msg.text}</ReactMarkdown>  // Bot Message
              )}
          </div>
        ))}

      {loading && (
        <div className="loading-indicator">
          <p>Loading...</p>
        </div>
      )}
        <div className='refresh-button'>
          <button onClick={handleRefresh}>Clear</button>
        </div>
      </div> 
      </>
      )}
      <div className="input-area">
        <input 
           type="text"
           value={input}
           placeholder="Ask your question..."
           onChange={(e) => setInput(e.target.value)}
           onKeyDown={(e) => e.key ==="Enter" && sendMessage()}
           />
           <button onClick={sendMessage}>Send</button>
      </div>
      </div>
    </div>
      </>
  )
}