"use client";

import React, { useState, useRef, useEffect } from "react";

import {
  ReactFlow,
  Background,
} from '@xyflow/react';
 
import '@xyflow/react/dist/style.css';

const nodeOrigin = [200, 20]

const MessageList = ({className, messages}) => {
	const messageEndRef = useRef(null);

	useEffect(() => {
		messageEndRef.current?.scrollIntoView({behavior: 'smooth'});
	}, [messages])

	return (
		<div className={className}>
			{messages.map((message, idx) => {
				return (
					<div key={idx} className={(message.from === 'user' ? "bg-gray-400" : "bg-blue-400") + " rounded shadow-lg p-2 mb-2"}
						dangerouslySetInnerHTML={{ __html: message.text }}>
					</div>
				);
			})}
			<div ref={messageEndRef} />
		</div>
	);
}

export default function Home() {

	const [graph, setGraph] = useState({nodes: [], edges: []});
	const [messages, setMessages] = useState([]);
	const [chatMessage, setChatMessage] = useState('');
	const textareaRef = useRef(null);

	useEffect(() => {
		if (textareaRef.current) {
		  textareaRef.current.style.height = 'auto';
		  textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
		}
	}, [chatMessage]);

	const handleSendMessage = () => {
		const requestOptions = {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({'category': chatMessage})
        };
        fetch('http://127.0.0.1:5000/api/get_graph', requestOptions)
        .then(response => response.json())
        .then(data => {
			const nodes = data.nodes
			const edges = data.edges
			updateGraph(nodes, edges)
			setMessages([...messages, {from: 'user', text: chatMessage}, {from: 'api', text: data.message}])
			setChatMessage('');
        });
	}

	const handleKeyDown = (event) => {
		if (event.key === 'Enter' && !event.shiftKey) {
			event.preventDefault();
			handleSendMessage();
		} else if (event.key === 'Enter' && !event.shiftKey) {
			setChatMessage(chatMessage + '\n');
		}
	}

	const updateGraph = (nodes, edges) => {
		const updatedNodes = nodes.flatMap((row, rowIndex) =>
			row.map((nodeData, colIndex) => ({
			  id: String(nodeData.display),
			  data: { label: String(nodeData.display) },
			  position: { x: nodeOrigin[0] + colIndex * 200, y: nodeOrigin[1] + rowIndex * 150 }, // Space out nodes
			  style: { backgroundColor: nodeData.type === "course" ? "lightblue" : "green", padding: "10px", borderRadius: "5px", color: 'black', fontSize: '14px' },
			}))
		);

		const updatedEdges = edges.map(({ from, to }) => ({
			id: `${from}-${to}`,
			source: String(from),
			target: String(to),
			animated: false,
			style: { stroke: "gray" },
		}));
		setGraph({nodes: updatedNodes, edges: updatedEdges});
	}

	return (
		<div className="grid grid-cols-2 gap-4 p-8 min-h-screen">
			<div className="flex-grow">
				<ReactFlow 
					nodes={graph.nodes} 
					edges={graph.edges}
				> 
					<Background />
				</ReactFlow>
			</div>

			<div className="bg-gray-100 flex flex-col p-4 rounded h-[730px]">
				<p className="text-lg text-gray-800 mb-2">Chat</p>
				<MessageList className="flex-grow overflow-y-auto p-2" messages={messages} />
				<div className="flex gap-2 mt-2"> 
					<textarea
						className="border p-2 w-full resize-none rounded shadow-xl text-gray-700"
						ref={textareaRef}
						value={chatMessage}
						placeholder="Type your message here..."
						onChange={(e) => setChatMessage(e.target.value)}
						onKeyDown={handleKeyDown}
						style={{ overflowY: 'hidden', resize: 'none' }}
					/>
					<button className="text-white end-2.5 bottom-2.5 bg-blue-700 hover:bg-blue-800 focus:ring-4 focus:outline-none focus:ring-blue-300 font-medium rounded-lg text-sm px-4 py-2" onClick={handleSendMessage}>Enter</button>
				</div>
			</div>
		</div>
  	);
}
