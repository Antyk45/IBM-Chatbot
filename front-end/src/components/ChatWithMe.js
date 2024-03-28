import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Bot from '../assets/bot.png';
import User from '../assets/user.png';
import ExampleQuestions from '../assets/ExampleQuestions.png'; // Import the ExampleQuestions.png image
import RatingLLms from './RatingLLms.js';
import './ChatWithMe.css'; // Import the CSS file

const mock = [{"answer":"\nAmazon Kindle E-Reader 6\" Wifi (8th Generation, 2016)","llm":"LLaMa"}, {"answer":"\nAmazon Kindle E-Reader 6\" Wifi (8th Generation, 2016)","llm":"AI21"}];

import io from 'socket.io-client'
const socket = io('http://localhost:5000');
const validLLMs = ["llama", "chatgpt", "ai21"];

const Bubble = ({ sender, message, className, ...props }) =>
<div className={`message ${sender} ${className ?? ""}`} {...props}>
    <text>{message}</text>
    <img src={sender === 'user' ? User : Bot} alt={`${sender} icon`} className="user-icon" />
</div>

const Answer = ({ type, ...rest }) => {
    if (type == "regular") {
        return rest["answer"];
    } else if (type == "tabular") {
        return <table>
                   <tr>
                       {rest.column_names.map(name => <td><span>{name}</span></td>)}
                   </tr>
                   {rest.results.map(result => <tr>{result.map(r => <td><span>{r}</span></td>)}</tr>)}
               </table>
    } else {
        return rest.answer?.trim() ?? <span style={{color: "red"}}>No data!</span>;
    }
}

function ChatWithMe() {
    axios.defaults.baseURL = 'http://localhost:5000';
    const [message, setMessage] = useState('');
    const [chatHistory, setChatHistory] = useState([]);
    const [answers, setAnswers] = useState({});
    const [sources, setSources] = useState([]);
    const [processingUserMessage, setProcessingUserMessage] = useState(false);
    const [llmRatings, setLlmRatings] = useState({});
    const [showImage, setShowImage] = useState(true); // New state for showing image

    useEffect(() => {
        socket.on("socket", (data) => {
            const { llm, answer } = data;
            setAnswers(answers => {
                const last = { ...answers, [llm]: { llm, answer } };
                console.log(last);
                return last;
            });
        });
    }, []);

    const handleInputChange = (event) => {
      setMessage(event.target.value);
    }

    const handleSendMessage = async (event) => {
        event.preventDefault();
        setProcessingUserMessage(() => true);
        setShowImage(false); // Hide image when sending message
        try {
            if (Object.keys(answers).length === 0) {
                // no LLM response to select, proceed.
            } else {
                const llm = document.querySelector('input[name="g1"]:checked')?.value;
                if (llm === undefined) {
                    throw Error("Not proceeding without llm chosen");
                } else {
                    try {
                        await axios.post('/selectAnswer', { llm })
                    } catch(error) {
                        console.error('Error selecting answer:', error);
                        return;
                    }
                    setChatHistory(chatHistory => [...chatHistory, { sender: 'bot', message: answers[llm].answer }]);
                    setAnswers(answers => ({}));
                }
            }

            const llms = [...document.querySelectorAll('[name="g2"]:checked')].map((input) => input.value);
            const sql = document.querySelector('#sql-checkbox').checked;

            await axios.post('/message', {message: message, llms, sql })
                .then(response => {
                    setChatHistory(chatHistory => [...chatHistory, { sender: 'user', message: message }]);
                    setAnswers(Object.fromEntries(response.data.answers.map(answer => [answer.llm, answer])));
                    setSources(response.data.sources);
                })
                .catch(error => {
                    console.error('Error fetching data:', error);
                });
        } catch (e) {
            if (e instanceof Error) alert(e);
        }
        setProcessingUserMessage(() => false);
    };

    const handleRatingChange = (llm, rating) => {
        setLlmRatings(prevState => ({
          ...prevState,
          [llm]: rating
        }));
    };
    
    return (
        <div className="master">
            <div className="options-pane">
                <div>
                    <h2>Enabled LLMs</h2>
                    <p>These LLMs will be included in the response</p>
                    {validLLMs.map((ai, index) => (
                        <div key={`enabled-llm.${ai}.${index}`}>
                            <label><input value={ai} type="checkbox" name="g2"/> {ai}</label>
                            <br/>
                        </div>
                    ))}
                    <br/>

                    <h2>Enabled Tools</h2>
                    <p>These tools will be used in the LLM response</p>
                        <div>
                            <label>
                                <input value="sqla" id="sql-checkbox" type="checkbox"/>Table Aggregation Tool - <small>If a question benefits from aggregations over the whole database, the agent will attempt to perform such an aggregation and return a table representing the result.</small>
                            </label>
                            <br/>
                        </div>
                    <br/>
                </div>
            </div>
            <div className="messages-pane">
                {showImage && <img src={ExampleQuestions} alt="Example Questions" />} {/* Image shown before message sent */}
                <div style={{flex: "1 1 0", overflow: "scroll"}}>
                    <div className="message-bubble">
                        { chatHistory.map((msg, index) => <Bubble {...msg}/>) }
                        { processingUserMessage && <Bubble className="loading" sender="user" message={message}/> }
                    </div>
                    {
                        sources.length > 0 && (
                            <>
                                <h3>Sources</h3>
                                { sources.map((source, index) => (<>
                                    <p><strong>Product:</strong> {source.productName}<br/>
                                    <strong>Rating:</strong> {source.rating ?? "N/A"}</p>
                                    <blockquote key={index}>{source.pageContent}</blockquote>
                                </>)) }
                            </>
                        )
                    }
                    {
                        (Object.keys(answers).length > 0) && (
                            <>
                                <h3>Answers</h3>
                                <center className="lmm-container-box">
                                    {
                                        Object.values(answers).map((msg, index) => (
                                            <div key={`answer.${index}.${msg.llm}`}
                                                 className={processingUserMessage
                                                            ? "processingUserMessage llm-container"
                                                            : "llm-container" }>
                                                <label>
                                                    <h4>{msg.llm}</h4>
                                                    <input defaultChecked={index == 0} value={msg.llm} id={"..."+index} type="radio" name="g1"/>
                                                    <span style={{whiteSpace: "pre-line"}}>{<Answer {...msg}/>}</span>
                                                </label>
                                                <div>
                                                    <RatingLLms llm={msg.llm} onRatingChange={handleRatingChange} />
                                                </div>
                                            </div>
                                        ))
                                    }
                                </center>
                            </>
                        )
                    }
                </div>
                <form className="input-form" onSubmit={handleSendMessage} >
                    <textarea placeholder="Ask your question here…" onChange={handleInputChange} />
                    {" "}
                    <input type="submit" disabled={processingUserMessage} value="→"/>
                </form>
            </div>
        </div>
    );
}

export default ChatWithMe;
