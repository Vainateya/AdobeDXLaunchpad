"use client";

import React, { useState, useRef, useEffect } from "react";
import { ReactFlow, Background } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { PulseLoader } from "react-spinners";

const nodeOrigin = [200, 20];

const MessageList = ({
	className,
	messages,
	messageRefs,
}: {
	className: string;
	messages: { from: string; text: string }[];
	messageRefs: React.RefObject<(HTMLDivElement | null)[]>;
}) => {
const messageEndRef = useRef(null);
useEffect(() => {
	messageEndRef.current?.scrollIntoView({ behavior: "smooth" });
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
			ref={(el) => {
				messageRefs.current[idx] = el;
			}}
			className={`rounded-lg p-4 shadow-md ${
			message.from === "user" ? "bg-[#EB1000]" : "bg-[#2A2A2A]"
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

useEffect(() => {
	if (textareaRef.current) {
	textareaRef.current.style.height = "auto";
	textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
	}
}, [chatMessage]);

const handleSendMessage = () => {
	if (!chatMessage.trim()) return;
	setLoading(true);

	const requestOptions = {
	method: "POST",
	headers: { "Content-Type": "application/json" },
	body: JSON.stringify({ category: chatMessage }),
	};

	fetch("http://127.0.0.1:5000/api/get_graph", requestOptions)
	.then((response) => response.json())
	.then((data) => {
		const nodes = data.nodes;
		const edges = data.edges;
		const userMsgIndex = messages.length;
		const newGraph = {
			nodes,
			edges,
			message: data.message,
			userMessage: chatMessage,
			messageIndex: userMsgIndex, // this tracks which message index to scroll to
		};

		setGraphHistory((prev) => [...prev, newGraph]);
		setCurrentGraphIndex(graphHistory.length);
		updateGraph(nodes, edges);
		setMessages([
		...messages,
		{ from: "user", text: chatMessage },
		{ from: "api", text: data.message },
		]);
		setChatMessage("");
	})
	.catch((error) => console.error("Error fetching data:", error))
	.finally(() => setLoading(false));
};

const handleKeyDown = (event) => {
	if (event.key === "Enter" && !event.shiftKey) {
	event.preventDefault();
	handleSendMessage();
	}
};

const updateGraph = (nodes, edges) => {
	const updatedNodes = nodes.flatMap((row, rowIndex) =>
	row.map((nodeData, colIndex) => ({
		id: String(nodeData.display),
		data: {
		...nodeData,
		label: `${nodeData.display} (${
			nodeData.type === "course" ? "Course" : "Certification"
		})`,
		},
		position: {
		x: nodeOrigin[0] + colIndex * 200,
		y: nodeOrigin[1] + rowIndex * 150,
		},
		style: {
		backgroundColor:
			nodeData.type === "course" ? "#EB1000" : "#2A2A2A",
		padding: "10px",
		borderRadius: "5px",
		color: "white",
		fontSize: "14px",
		fontWeight: "bold",
		textAlign: "center",
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
		History
	</button>
	)}
	{/* Bottom-left Survey Button */}
	<button
		onClick={() => setShowSurvey(true)}
		className="fixed bottom-6 left-6 bg-[#EB1000] hover:bg-red-700 text-white px-4 py-2 rounded-full shadow-lg z-50"
		>
		Answer a Few Questions
	</button>

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
			1. Are you already certified?
			</label>
			<div className="flex gap-4">
			<button
				className={`px-4 py-2 rounded ${
				isCertified === "yes" ? "bg-[#EB1000]" : "bg-[#333]"
				}`}
				onClick={() => setIsCertified("yes")}
			>
				Yes
			</button>
			<button
				className={`px-4 py-2 rounded ${
				isCertified === "no" ? "bg-[#EB1000]" : "bg-[#333]"
				}`}
				onClick={() => setIsCertified("no")}
			>
				No
			</button>
			</div>
		</div>

		{/* Question 2 */}
		<div className="mb-4">
			<label className="block text-sm font-medium mb-2">
			2. Years of experience in your field:
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
				certified: isCertified,
				experience_years: experienceYears,
				};
			
				fetch("http://127.0.0.1:5000/api/survey", {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify(surveyData),
				})
				.then((res) => res.json())
				.then((data) => {
					console.log("Survey submitted:", data);
				})
				.catch((err) => console.error("Survey submission failed:", err))
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
					const target = messageRefs.current[item.messageIndex];
					if (target) {
						target.scrollIntoView({ behavior: "smooth", block: "start" });
					}
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
		<div
			className="flex-grow"
			onClick={() => setShowHistory(false)}
		/>
		</div>
	)}

	{/* Graph Display */}
	<div className="flex-grow relative">
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
			<p className="mb-2 text-sm font-bold uppercase tracking-wide text-gray-300">
			{hoveredNode.data.type === "course"
				? "Course"
				: "Certification"}
			</p>

			{hoveredNode.data.data?.category && (
			<p className="text-sm mb-1">
				<strong>Category:</strong>{" "}
				{hoveredNode.data.data.category}
			</p>
			)}
			{hoveredNode.data.data?.level && (
			<p className="text-sm mb-1">
				<strong>Level:</strong> {hoveredNode.data.data.level}
			</p>
			)}
			{hoveredNode.data.data?.job_role && (
			<p className="text-sm mb-1">
				<strong>Job Role:</strong>{" "}
				{hoveredNode.data.data.job_role}
			</p>
			)}

			{hoveredNode.data.type === "course" &&
			hoveredNode.data.data?.objectives && (
				<p className="text-sm mb-1 whitespace-pre-wrap">
				<strong>Objectives:</strong>{" "}
				{hoveredNode.data.data.objectives}
				</p>
			)}

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

		<MessageList
			className="flex-grow overflow-y-auto p-2"
			messages={messages}
			messageRefs={messageRefs}
		/>


		{loading && (
		<div className="flex justify-center my-2">
			<PulseLoader color="#EB1000" />
		</div>
		)}

		<div className="flex gap-2 mt-4">
		<textarea
			className="border p-3 w-full resize-none rounded-lg shadow-xl text-black bg-white text-lg placeholder-gray-500"
			ref={textareaRef}
			value={chatMessage}
			placeholder="Type your question here..."
			onChange={(e) => setChatMessage(e.target.value)}
			onKeyDown={handleKeyDown}
			style={{ overflowY: "hidden", resize: "none" }}
		/>
		<button
			className={`text-white font-medium text-lg px-5 py-3 transition-all rounded-lg ${
			loading
				? "bg-gray-500 cursor-not-allowed"
				: "bg-[#EB1000] hover:bg-red-700"
			}`}
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
