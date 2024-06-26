import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
import InsertDocument from './components/InsertDocument';
import ChatWithMe from './components/ChatWithMe';
import AboutUs from './components/AboutUs';
import 'bootstrap/dist/css/bootstrap.css';
import './App.css';
import { ReactComponent as Logo } from './logo.svg'; // Ensure the path is correct
import InsertApiKey from './components/InsertApiKey';
import InsertAI21API from './components/InsertAI21API';
import LandingPage from './components/LandingPage';


function App() {

  return (
    <Router>
      <header className="App-header">
        <Logo className="App-logo" alt="logo" /> {/* Logo added here */}
                <nav className="nav nav-masthead justify-content-center">
                  <Link className="nav-link" to="/">Home</Link>
                  <Link className="nav-link" to="/insert-document">Insert Document</Link>
                  <Link className="nav-link" to="/chat-with-me">Chat With Me</Link>
                  <div className="dropdown">
                  <span className="nav-link">Insert API Keys</span>
                    <div className="dropdown-content">
                      <Link className="nav-link" to="/openai-api-key">OpenAI API Key</Link>
                      <Link className="nav-link" to="/ai21-api-key">AI21 API Key</Link>
                    </div>
                  </div>
                  <Link className="nav-link" to="/about-us">About Us</Link>
                </nav>
      </header>
        <div className="container-fluid d-flex justify-content-center align-items-stretch app-diagonal-gradient main-parent">
            <main role="main" className="inner cover">
              <Routes>
                <Route className= "home-page-1" path="/" element={<LandingPage />} />
                <Route path="/insert-document" element={<InsertDocument />} />
                <Route path="/chat-with-me" element={<ChatWithMe />} />
                <Route className = "insert-api-key-123" path="/openai-api-key" element={<InsertApiKey/>} />
                <Route className = "insert-api-key-123" path="/ai21-api-key" element={<InsertAI21API/>} />
                <Route path="/about-us" element={<AboutUs />} />
              </Routes>
            </main>
          </div>
    </Router>
  );
}

export default App;
