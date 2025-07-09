"use client";

import React, { useState, useRef, useEffect } from "react";
import { ReactFlow, Background } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { PulseLoader } from "react-spinners";
import "./glowButton.css";

const nodeOrigin = [200, 20];

//const host = "ec2-18-218-177-90.us-east-2.compute.amazonaws.com";
const host = "127.0.0.1";

const character_limit = 140;

const MessageList = ({
  className,
  messages,
  messageRefs,
}: {
  className: string;
  messages: { from: string; text: string }[];
  messageRefs: React.RefObject<(HTMLDivElement | null)[]>;
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const messageEndRef = useRef<HTMLDivElement>(null);
  const [isAutoScroll, setIsAutoScroll] = useState(true);

  // Handle scroll to bottom based on flag
  useEffect(() => {
    if (isAutoScroll) {
      messageEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, isAutoScroll]);

  // Monitor scroll position
  const handleScroll = () => {
    const container = containerRef.current;
    if (!container) return;

    const isAtBottom =
      container.scrollHeight - container.scrollTop <=
      container.clientHeight + 10;
    setIsAutoScroll(isAtBottom);
  };

  return (
    <div
      className={`${className} text-white overflow-y-auto`}
      ref={containerRef}
      onScroll={handleScroll}
    >
      {messages.length === 0 ? (
        <>
          <div className="text-gray-400 text-sm italic">
            Try asking:
            <ul className="list-disc ml-4 mt-2">
              <li>What Adobe courses should I take?</li>
              <li>Show me a learning path for Adobe Analytics.</li>
              <li>How do I become an Adobe Certified Expert?</li>
            </ul>
          </div>
        </>
      ) : (
        messages.map((message, idx) => (
          <div
            key={idx}
            ref={(el) => {
              messageRefs.current[idx] = el;
            }}
            className={`rounded-lg p-4 shadow-md mb-2 ${
              message.from === "user" ? "bg-[#FF0000]" : "bg-[#2A2A2A]"
            }`}
            style={{ color: "#FFFFFF" }}
          >
            <div
              className="prose prose-invert"
              style={{ color: "#FFFFFF" }}
              dangerouslySetInnerHTML={{ __html: message.text }}
            />
          </div>
        ))
      )}
      <div ref={messageEndRef} />
    </div>
  );
};

export default function Home() {
  const [graph, setGraph] = useState({ nodes: [], edges: [] });
  const [messages, setMessages] = useState([]);
  const [chatMessage, setChatMessage] = useState("");
  const [loading, setLoading] = useState(false);
  const textareaRef = useRef(null);
  const [hoveredNode, setHoveredNode] = useState(null);
  const [hoverPosition, setHoverPosition] = useState({ x: 0, y: 0 });
  const messageRefs = useRef<(HTMLDivElement | null)[]>([]);

  const [graphHistory, setGraphHistory] = useState([]);
  const [currentGraphIndex, setCurrentGraphIndex] = useState(null);
  const [showHistory, setShowHistory] = useState(false);
  const [showSurvey, setShowSurvey] = useState(false);
  const [isCertified, setIsCertified] = useState(null); // yes or no
  const [experienceYears, setExperienceYears] = useState("");
  // const [glowActive, setGlowActive] = useState(true);
  const [certificationText, setCertificationText] = useState("");
  const [areaOfExpertise, setAreaOfExpertise] = useState("");
  const [graphEnabled, setGraphEnabled] = useState(false);
  const questionPrompts = [
    "Tell me about Adobe's ",
    "What does Adobe offer in ",
    "Which certification should I get for ",
    "What courses are best for learning ",
  ];
  const [showPrompts, setShowPrompts] = useState(false);
  const [allResourceNames, setAllResourceNames] = useState<string[]>([]);
  const [allCategories, setAllCategories] = useState<string[]>([]);
  const [selectedCerts, setSelectedCerts] = useState<string[]>([]);
  const [selectedCategory, setSelectedCategory] = useState<string>("");
  const [customCategory, setCustomCategory] = useState<string>("");

  useEffect(() => {
    // Fetch resources.json to get all course/cert names and categories
    fetch("/resources.json")
      .then((res) => res.json())
      .then((data) => {
        const names = Object.keys(data);
        setAllResourceNames(names);
        const categories = Array.from(
          new Set(Object.values(data).map((item: any) => item.category))
        );
        setAllCategories(categories);
      })
      .catch(() => {
        setAllResourceNames([]);
        setAllCategories([]);
      });
  }, []);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  }, [chatMessage]);
  // useEffect(() => {
  //   const timer = setTimeout(() => setGlowActive(false), 5000);
  //   return () => clearTimeout(timer);
  // }, []);
  const updateGraphAndStreamResponse = async () => {
    if (!chatMessage.trim()) return;
    setLoading(true);
    const userMessage = chatMessage;
    setMessages((prev) => [...prev, { from: "user", text: userMessage }]);
    setLengthRestrictedChatMessage("");

    try {
      const streamResponse = await fetch(
        "http://" + host + ":5000/api/stream_response",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            category: userMessage,
            graph_enabled: graphEnabled,
          }),
        }
      );

      const reader = streamResponse.body?.getReader();
      const decoder = new TextDecoder();
      let result = "";

      const newMessage = { from: "api", text: "" };
      setMessages((prev) => [...prev, newMessage]);

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        result += chunk;

        setMessages((prev) => {
          const updated = [...prev];
          updated[updated.length - 1] = {
            ...updated[updated.length - 1],
            text: result,
          };
          return updated;
        });
      }

      // Now that streaming is complete, generate the graph if enabled
      if (graphEnabled) {
        try {
          console.log("Generating graph after response completion...");
          const graphResponse = await fetch(
            "http://" + host + ":5000/api/get_graph_from_response",
            {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({}), // No body needed
            }
          );

          const graphData = await graphResponse.json();
          console.log("Graph response:", graphData);

          if (graphData.nodes && graphData.nodes.length > 0) {
            updateGraph(graphData.nodes, graphData.edges);

            const userMsgIndex = messages.length;
            const newGraph = {
              nodes: graphData.nodes,
              edges: graphData.edges,
              message: graphData.message || "",
              userMessage: userMessage,
              messageIndex: userMsgIndex,
            };
            setGraphHistory((prev) => [...prev, newGraph]);
            setCurrentGraphIndex(graphHistory.length);
          } else {
            console.log("No graph generated:", graphData.message);
          }
        } catch (err) {
          console.error("Error generating graph:", err);
        }
      }

      setLoading(false);
    } catch (err) {
      console.error("Error streaming response:", err);
      setLoading(false);
    }
  };

  const setLengthRestrictedChatMessage = (message: string) => {
    if (message.length <= character_limit) {
      setChatMessage(message);
    }
  };

  const handleSendMessage = () => {
    updateGraphAndStreamResponse();
  };

  const handleKeyDown = (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      handleSendMessage();
    }
  };

  const getNodeColor = (type: string) => {
    if (type === "course") {
      return "#FF0000";
    } else if (type === "certificate") {
      return "#2A2A2A";
    } else {
      return "#1151B8";
    }
  };

  const updateGraph = (nodes, edges) => {
    const updatedNodes = nodes.flatMap((row, rowIndex) =>
      row.map((nodeData, colIndex) => ({
        id: String(nodeData.display),
        data: {
          ...nodeData, // includes type, display, data
          label:
            nodeData.type === "course" || nodeData.type === "certificate"
              ? `${nodeData.display} (${
                  nodeData.type === "course" ? "Course" : "Certification"
                })`
              : nodeData.display,
        },
        position: {
          x: nodeOrigin[0] + colIndex * 200,
          y: nodeOrigin[1] + rowIndex * 150,
        },
        style: {
          backgroundColor: getNodeColor(nodeData.type),
          padding: "10px",
          borderRadius: "5px",
          color: "white",
          fontSize: "14px",
          fontWeight: "bold",
        },
      }))
    );

    const updatedEdges = edges.map(({ from, to }) => ({
      id: `${from}-${to}`,
      source: String(from),
      target: String(to),
      animated: false,
      style: { stroke: "gray" },
    }));

    setGraph({ nodes: updatedNodes, edges: updatedEdges });
  };

  return (
    <div className="grid grid-cols-2 gap-4 p-8 min-h-screen font-sans relative">
      {/* History Button */}
      {!showHistory && (
        <button
          onClick={() => setShowHistory(true)}
          className="absolute top-10 left-4 bg-[#EB1000] hover:bg-red-700 text-white font-semibold py-2 px-4 rounded shadow-lg z-50"
        >
          Graph History
        </button>
      )}
      {/* Bottom-left Survey Button */}
      {/* <button
        onClick={() => setShowSurvey(true)}
        className="fixed bottom-6 left-6 bg-[#EB1000] hover:bg-red-700 text-white px-4 py-2 rounded-full shadow-lg z-50"
      >
        Answer a Few Questions
      </button> */}

      {showSurvey && (
        <div className="fixed inset-0 z-50 bg-black bg-opacity-50 flex items-center justify-center">
          <div className="bg-[#1f1f1f] rounded-xl p-6 w-[400px] max-w-full shadow-2xl text-white relative">
            <button
              className="absolute top-2 right-3 text-gray-300 hover:text-white text-lg"
              onClick={() => setShowSurvey(false)}
            >
              ✕
            </button>
            <h2 className="text-xl font-semibold mb-4">Quick Questions</h2>

            {/* Question 1 */}
            <div className="mb-4">
              <label className="block text-sm font-medium mb-2">
                1. What Adobe course or certification(s) did you do?
              </label>
              <div className="flex flex-wrap gap-2 mb-2">
                {selectedCerts.map((cert) => (
                  <span
                    key={cert}
                    className="bg-red-600 text-white px-2 py-1 rounded flex items-center"
                  >
                    {cert}
                    <button
                      className="ml-2 text-white hover:text-gray-200"
                      onClick={() =>
                        setSelectedCerts(
                          selectedCerts.filter((c) => c !== cert)
                        )
                      }
                      aria-label={`Remove ${cert}`}
                      type="button"
                    >
                      ×
                    </button>
                  </span>
                ))}
              </div>
              <select
                className="w-full px-4 py-2 border border-gray-300 rounded bg-white text-black"
                value=""
                onChange={(e) => {
                  const value = e.target.value;
                  if (value && !selectedCerts.includes(value)) {
                    setSelectedCerts([...selectedCerts, value]);
                  }
                }}
              >
                <option value="">Select a course or certification...</option>
                {allResourceNames
                  .filter((name) => !selectedCerts.includes(name))
                  .map((name) => (
                    <option key={name} value={name}>
                      {name}
                    </option>
                  ))}
              </select>
            </div>

            {/* Question 2 */}
            <div className="mb-4">
              <label className="block text-sm font-medium mb-2">
                2. Solution of interest:
              </label>
              <select
                className="w-full px-4 py-2 border border-gray-300 rounded bg-white text-black"
                value={selectedCategory}
                onChange={(e) => setSelectedCategory(e.target.value)}
              >
                <option value="">Select a category</option>
                {allCategories.length === 0 ? (
                  <option disabled>No categories found</option>
                ) : (
                  allCategories.map((cat) => (
                    <option key={cat} value={cat}>
                      {cat}
                    </option>
                  ))
                )}
                <option value="Other">Other</option>
              </select>
              {selectedCategory === "Other" && (
                <input
                  type="text"
                  className="w-full px-4 py-2 border border-gray-300 rounded bg-white text-black mt-2"
                  value={customCategory}
                  onChange={(e) => setCustomCategory(e.target.value)}
                  placeholder="Enter your solution of interest"
                />
              )}
            </div>

            {/* Question 3 */}
            <div className="mb-4">
              <label className="block text-sm font-medium mb-2">
                3. Years of experience in your field:
              </label>
              <input
                type="number"
                min="0"
                value={experienceYears}
                onChange={(e) => setExperienceYears(e.target.value)}
                className="w-full px-3 py-2 rounded bg-white text-black"
                placeholder="e.g., 3"
              />
            </div>

            {/* Submit (optional) */}
            <button
              className="mt-2 bg-[#EB1000] hover:bg-red-700 text-white px-4 py-2 rounded"
              onClick={() => {
                const surveyData = {
                  certifications: selectedCerts,
                  expertise:
                    selectedCategory === "Other"
                      ? customCategory
                      : selectedCategory,
                  experience_years: experienceYears,
                };

                fetch("http://" + host + ":5000/api/survey", {
                  method: "POST",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify(surveyData),
                })
                  .then((res) => res.json())
                  .then((data) => {
                    console.log("Survey submitted:", data);
                  })
                  .catch((err) =>
                    console.error("Survey submission failed:", err)
                  )
                  .finally(() => setShowSurvey(false));
              }}
            >
              Submit
            </button>
          </div>
        </div>
      )}

      {/* History Sidebar */}
      {showHistory && (
        <div className="fixed inset-0 z-40 bg-black bg-opacity-50 flex">
          <div className="w-80 bg-[#1f1f1f] p-4 overflow-y-auto shadow-2xl">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-white text-lg font-bold">Graph History</h2>
              <button
                className="text-gray-300 hover:text-white text-xl"
                onClick={() => setShowHistory(false)}
              >
                ✕
              </button>
            </div>

            {graphHistory.length === 0 ? (
              <p className="text-sm text-gray-400 italic">No history yet.</p>
            ) : (
              graphHistory.map((item, index) => (
                <div
                  key={index}
                  onClick={() => {
                    updateGraph(item.nodes, item.edges);
                    setCurrentGraphIndex(index);
                    setShowHistory(false);

                    // Scroll to corresponding chat message
                    const target = messageRefs.current[item.messageIndex];
                    if (target) {
                      target.scrollIntoView({
                        behavior: "smooth",
                        block: "start",
                      });
                    }

                    // Fetch current graph items from backend
                    // Send current nodes to backend
                    console.log(" Sending nodes to backend:", item.nodes);
                    console.log(
                      " Updated graph items sent to backend:",
                      item.nodes
                    );
                    fetch("http://" + host + ":5000/api/set_current_graph", {
                      method: "POST",
                      headers: { "Content-Type": "application/json" },
                      body: JSON.stringify({ nodes: item.nodes }),
                    })
                      .then((res) => res.json())
                      .then((data) => {
                        console.log(
                          " Updated graph items sent to backend:",
                          data.current_items
                        );
                      })
                      .catch((err) =>
                        console.error("Error sending graph items:", err)
                      );
                  }}
                  className={`text-white text-sm p-3 rounded mb-2 cursor-pointer ${
                    index === currentGraphIndex
                      ? "bg-[#EB1000]"
                      : "bg-[#2A2A2A] hover:bg-[#3A3A3A]"
                  }`}
                >
                  <p className="font-semibold truncate">{item.userMessage}</p>
                  <p className="text-xs italic text-gray-300">
                    {item.message.slice(0, 60)}...
                  </p>
                </div>
              ))
            )}
          </div>
          <div className="flex-grow" onClick={() => setShowHistory(false)} />
        </div>
      )}

      {/* Graph Display */}
      <div className="flex-grow relative">
        {/* Toggle Button (Enable/Disable Graph Generation) */}
        <div className="absolute top-4 right-4 z-50 flex items-center gap-3">
          <span className="text-base font-semibold text-black dark:text-white">
            Enable Graph
          </span>
          <label className="relative inline-flex items-center cursor-pointer">
            <input
              type="checkbox"
              className="sr-only peer"
              checked={graphEnabled}
              onChange={() => setGraphEnabled((prev) => !prev)}
            />
            <div className="w-11 h-6 bg-gray-300 dark:bg-gray-600 rounded-full peer peer-checked:bg-red-600 transition-colors duration-300" />
            <div className="absolute left-0.5 top-0.5 w-5 h-5 bg-white dark:bg-gray-100 rounded-full shadow-md transform peer-checked:translate-x-full transition-transform duration-300" />
          </label>
        </div>

        <ReactFlow
          nodes={graph.nodes}
          edges={graph.edges}
          onNodeMouseEnter={(event, node) => {
            const target = event.target as HTMLElement;
            const boundingRect = target.getBoundingClientRect();
            setHoverPosition({
              x: boundingRect.right,
              y: boundingRect.top + boundingRect.height / 2 - 40,
            });
            setHoveredNode(node);
          }}
          onNodeMouseLeave={() => setHoveredNode(null)}
        >
          <Background />
        </ReactFlow>

        {/* Hover Info Box */}
        {hoveredNode && (
          <div
            style={{
              position: "fixed",
              left: hoverPosition.x,
              top: hoverPosition.y,
              width: "300px",
              backgroundColor: "#333",
              color: "#fff",
              padding: "15px",
              borderRadius: "8px",
              boxShadow: "0 4px 8px rgba(0,0,0,0.3)",
              zIndex: 100,
              maxHeight: "60vh",
              overflowY: "auto",
            }}
            onMouseEnter={() => setHoveredNode(hoveredNode)}
            onMouseLeave={() => setHoveredNode(null)}
          >
            {/* Title */}
            <p className="mb-2 text-sm font-bold uppercase tracking-wide text-gray-300">
              {hoveredNode.data.type}
            </p>

            {/* Shared Fields */}
            {hoveredNode.data.data?.category && (
              <p className="text-sm mb-1">
                <strong>Category:</strong> {hoveredNode.data.data.category}
              </p>
            )}
            {hoveredNode.data.data?.level && (
              <p className="text-sm mb-1">
                <strong>Level:</strong> {hoveredNode.data.data.level}
              </p>
            )}
            {hoveredNode.data.data?.job_role && (
              <p className="text-sm mb-1">
                <strong>Job Role:</strong> {hoveredNode.data.data.job_role}
              </p>
            )}

            {/* Course-specific: only show objectives */}
            {hoveredNode.data.type === "course" &&
              hoveredNode.data.data?.objectives && (
                <p className="text-sm mb-1 whitespace-pre-wrap">
                  <strong>Objectives:</strong>{" "}
                  {hoveredNode.data.data.objectives}
                </p>
              )}

            {/* Certificate-specific: no additional fields */}

            {/* Link */}
            {hoveredNode.data.data?.link ? (
              <a
                href={hoveredNode.data.data.link}
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-400 underline text-sm mt-2 inline-block"
              >
                Visit Official Page →
              </a>
            ) : (
              <p className="text-gray-400 text-sm italic mt-2">
                No link available
              </p>
            )}
          </div>
        )}
      </div>

      {/* Chat Interface */}
      <div className="bg-[#2A2A2A] flex flex-col p-6 rounded-xl h-[calc(100vh-32px)] shadow-lg">
        <div className="flex justify-between items-center mb-4">
          <p className="text-2xl font-bold text-white">Adobe Chat</p>
        </div>

        {/* Scrollable chat area including instructions and messages */}
        <div className="flex-grow overflow-y-auto">
          <div className="mb-4 p-4 bg-[#222] rounded-lg text-white text-sm shadow">
            <strong>Welcome!</strong>
            <p>
              This is an AI-powered Adobe learning and certification assistant.
              Ask questions about Adobe courses, certifications, and learning
              paths. The tool uses advanced AI to find the most relevant
              resources and can visualize learning pathways as interactive
              graphs.
            </p>
            <ul className="list-disc ml-5 mt-2">
              <li>
                <b>Best for:</b> Exploring Adobe certifications, finding course
                prerequisites, and planning your learning journey.
              </li>
              <li>
                <b>Not for:</b> Real-time exam registration, official Adobe
                support, or non-Adobe topics.
              </li>
              <li>
                <b>Interface tips:</b> Toggle the graph for visual learning
                paths. Use “Answer a few questions” to get personalized
                recommendations.
              </li>
            </ul>
          </div>
          <MessageList
            className="flex-grow overflow-y-auto p-2"
            messages={messages}
            messageRefs={messageRefs}
          />
        </div>

        {loading && (
          <div className="flex justify-center my-2">
            <PulseLoader color="#FF0000" />
          </div>
        )}
        {/* Quick Prompts Toggle Box */}
        <div className="mt-2 mb-4">
          <button
            className="text-sm text-white bg-[#333] hover:bg-[#444] px-4 py-2 rounded-lg mb-2 shadow-md"
            onClick={() => setShowPrompts((prev) => !prev)}
          >
            {showPrompts ? "Hide Quick Prompts" : "Show Quick Prompts"}
          </button>

          {showPrompts && (
            <div className="flex flex-col gap-3 mt-2">
              {questionPrompts.map((prompt, idx) => (
                <button
                  key={idx}
                  className="bg-red-500/20 backdrop-blur-md text-white px-4 py-3 rounded-lg text-base text-left shadow-md border border-red-400 hover:bg-red-500/30 transition duration-200"
                  onClick={() => setLengthRestrictedChatMessage(prompt)}
                >
                  {prompt}...
                </button>
              ))}
            </div>
          )}
        </div>

        <div>
          <div className="flex gap-2 mt-4">
            <textarea
              className="border p-3 w-full resize-none rounded-lg shadow-xl text-black bg-white text-lg placeholder-gray-500"
              ref={textareaRef}
              value={chatMessage}
              placeholder="Type your question here..."
              onChange={(e) => setLengthRestrictedChatMessage(e.target.value)}
              onKeyDown={handleKeyDown}
              style={{ overflowY: "hidden", resize: "none" }}
            />
            <button
              className={`text-white font-medium text-lg px-5 py-3 transition-all rounded-lg ${
                loading
                  ? "bg-gray-500 cursor-not-allowed"
                  : "bg-[#FF0000] hover:bg-red-700"
              }`}
              onClick={handleSendMessage}
              disabled={loading}
            >
              {loading ? "..." : "➤"}
            </button>
          </div>
          <p
            className={`text-sm mt-1 ml-1 ${
              chatMessage.length == character_limit ? "text-red-500" : ""
            }`}
          >
            {chatMessage.length}/{character_limit}
          </p>
        </div>
      </div>
      <button
        className="glow-button fixed hover:bg-red-700 bottom-6 left-6 z-50 wiggle-once"
        onClick={() => setShowSurvey(true)}
      >
        Answer a Few Questions
      </button>
    </div>
  );
}
