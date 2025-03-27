"use client";

import React, { useState, useRef, useEffect } from "react";
import { ReactFlow, Background } from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { PulseLoader } from "react-spinners"; 

const nodeOrigin = [200, 20];

const MessageList = ({ className, messages }) => {
	const messageEndRef = useRef(null);

	useEffect(() => {
		messageEndRef.current?.scrollIntoView({ behavior: 'smooth' });
	}, [messages]);

	return (
		<div className={`${className} text-white`}>
			{messages.length === 0 ? (
				<div className="text-gray-400 text-sm italic">
					Try asking:
					<ul className="list-disc ml-4 mt-2">
						<li>📌 &quot;What Adobe courses should I take?&quot;</li>
						<li>📌 &quot;Show me a learning path for Adobe Analytics.&quot;</li>
						<li>📌 &quot;How do I become an Adobe Certified Expert?&quot;</li>
					</ul>
				</div>
			) : (
				messages.map((message, idx) => (
					<div 
						key={idx} 
						className={`rounded-lg p-4 shadow-md ${
							message.from === 'user' ? "bg-[#FF0000]" : "bg-[#2A2A2A]"
						}`}
						style={{
							color: "#FFFFFF", // ✅ Forces text to be white
						}}
					>
						<div
							className="prose prose-invert" // ✅ Ensures all text is white
							style={{
								color: "#FFFFFF", 
							}}
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
	const [chatMessage, setChatMessage] = useState('');
	const [loading, setLoading] = useState(false);
	const textareaRef = useRef(null);

	useEffect(() => {
		if (textareaRef.current) {
			textareaRef.current.style.height = 'auto';
			textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
		}
	}, [chatMessage]);

	const handleSendMessage = () => {
		if (!chatMessage.trim()) return;
		setLoading(true);

		const requestOptions = {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ 'category': chatMessage })
		};

		fetch('http://127.0.0.1:5000/api/get_graph', requestOptions)
			.then(response => response.json())
			.then(data => {
				const nodes = data.nodes;
				const edges = data.edges;
				updateGraph(nodes, edges);
				setMessages([...messages, { from: 'user', text: chatMessage }, { from: 'api', text: data.message }]);
				setChatMessage('');
			})
			.catch(error => console.error("Error fetching data:", error))
			.finally(() => setLoading(false));
	};

	const handleKeyDown = (event) => {
		if (event.key === 'Enter' && !event.shiftKey) {
			event.preventDefault();
			handleSendMessage();
		}
	};

	const updateGraph = (nodes, edges) => {
		const updatedNodes = nodes.flatMap((row, rowIndex) =>
			row.map((nodeData, colIndex) => ({
				id: String(nodeData.display),
				data: { label: String(nodeData.display) },
				position: { x: nodeOrigin[0] + colIndex * 200, y: nodeOrigin[1] + rowIndex * 150 }, 
				style: { 
					backgroundColor: nodeData.type === "course" ? "#FF0000" : "#2A2A2A", 
					padding: "10px", 
					borderRadius: "5px", 
					color: 'white', 
					fontSize: '14px',
					fontWeight: "bold"
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
		<div className="grid grid-cols-2 gap-4 p-8 min-h-screen font-sans">
			{/* Graph Display */}
			<div className="flex-grow">
				<ReactFlow nodes={graph.nodes} edges={graph.edges}>
					<Background />
				</ReactFlow>
			</div>

			{/* Chat Interface */}
			<div className="bg-[#2A2A2A] flex flex-col p-6 rounded-xl h-[calc(100vh-32px)] shadow-lg">
				{/* Header with Dark Mode Toggle */}
				<div className="flex justify-between items-center mb-4">
					<p className="text-2xl font-bold text-white">Adobe Chat</p>
				</div>

				{/* Message List */}
				<MessageList className="flex-grow overflow-y-auto p-2" messages={messages} />

				{/* ✅ Show Loading Indicator */}
				{loading && (
					<div className="flex justify-center my-2">
						<PulseLoader color="#FF0000" />
					</div>
				)}

				{/* Chat Input */}
				<div className="flex gap-2 mt-4">
					<textarea
						className="border p-3 w-full resize-none rounded-lg shadow-xl text-black bg-white text-lg placeholder-gray-500"
						ref={textareaRef}
						value={chatMessage}
						placeholder="Type your question here..."
						onChange={(e) => setChatMessage(e.target.value)}
						onKeyDown={handleKeyDown}
						style={{ overflowY: 'hidden', resize: 'none' }}
					/>
					<button
						className={`text-white font-medium text-lg px-5 py-3 transition-all rounded-lg
									${loading ? "bg-gray-500 cursor-not-allowed" : "bg-[#FF0000] hover:bg-red-700"}`}
						onClick={handleSendMessage}
						disabled={loading}
					>
						{loading ? "..." : "➤"}
					</button>
				</div>
			</div>
		</div>
	);
}