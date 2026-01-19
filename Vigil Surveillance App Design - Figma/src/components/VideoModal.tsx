import { useEffect, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Download, Volume2, VolumeX, Play, Pause, WifiOff } from 'lucide-react';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { downloadVideoClip } from '../utils/exportUtils';
import { toast } from 'sonner';

import React from 'react';
interface VideoModalProps {
	isOpen: boolean;
	onClose: () => void;
	videoUrl?: string;
	offlineMode?: boolean;
	cameraId?: string;
	incidentType?: string;
	timestamp?: string;
	location?: string;
	controls?: React.ReactNode;
	showViewLiveButton?: boolean;
}

export function VideoModal({
	isOpen,
	onClose,
	videoUrl,
	offlineMode = false,
	cameraId,
	incidentType,
	timestamp,
	location,
	controls,
	showViewLiveButton = false,
}: VideoModalProps) {
	const [isMuted, setIsMuted] = useState(true);
	const [isPlaying, setIsPlaying] = useState(false);
	const [videoError, setVideoError] = useState(false);
	const videoRef = useRef<HTMLVideoElement>(null);

	useEffect(() => {
		if (!isOpen) {
			setIsPlaying(false);
			setVideoError(false);
		}
	}, [isOpen]);

	const handleBackdropClick = () => {
		onClose();
	};

	const toggleMute = () => {
		if (videoRef.current) {
			videoRef.current.muted = !isMuted;
			setIsMuted(!isMuted);
		}
	};

	const togglePlayPause = () => {
		if (videoRef.current) {
			if (isPlaying) {
				videoRef.current.pause();
			} else {
				videoRef.current.play();
			}
			setIsPlaying(!isPlaying);
		}
	};

	const handleDownload = async () => {
		try {
			await downloadVideoClip(videoUrl || '', cameraId || '', timestamp || new Date().toISOString());
			toast.success('Video downloaded successfully');
		} catch (error) {
			console.error('Download failed:', error);
			toast.error('Failed to download video');
		}
	};

	return (
		<AnimatePresence>
			{isOpen && (
				<motion.div
					initial={{ opacity: 0 }}
					animate={{ opacity: 1 }}
					exit={{ opacity: 0 }}
					transition={{ duration: 0.2 }}
					className="fixed inset-0 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4"
					style={{ zIndex: 2147483646, background: '#0a2540', pointerEvents: 'all' }}
					onClick={handleBackdropClick}
				>
					<motion.div
						initial={{ scale: 0.9, opacity: 0 }}
						animate={{ scale: 1, opacity: 1 }}
						exit={{ scale: 0.9, opacity: 0 }}
						transition={{ type: 'spring', damping: 25, stiffness: 300 }}
						className="relative w-full max-w-[min(1000px,90vw)] h-[min(700px,85vh)] bg-gradient-to-br from-gray-950 to-black border border-cyan-500/30 rounded-lg overflow-hidden shadow-2xl shadow-cyan-500/20"
						style={{ background: '#0a2540' }}
						onClick={(e) => e.stopPropagation()}
					>
						{/* Header */}
						<div className="absolute top-0 left-0 right-0 z-10 bg-gradient-to-b from-black/90 via-black/70 to-transparent backdrop-blur-md border-b border-cyan-500/20 p-4">
							<div className="flex items-center justify-between">
								<div className="flex items-center gap-3">
									<Badge className="bg-cyan-500/20 text-cyan-400 border-cyan-500/30">
										{cameraId}
									</Badge>
									{incidentType && (
										<Badge className={
											incidentType === 'violence'
												? 'bg-red-500/20 text-red-400 border-red-500/30'
												: 'bg-amber-500/20 text-amber-400 border-amber-500/30'
										}>
											{incidentType}
										</Badge>
									)}
									{timestamp && (
										<span className="text-sm text-gray-400 font-mono">
											{timestamp}
										</span>
									)}
								</div>
								<div className="flex items-center gap-2">
									{showViewLiveButton && (
										<Button size="sm" variant="default" className="bg-blue-600 text-white hover:bg-blue-700" onClick={() => window.open(videoUrl, '_blank')}>
											View Live
										</Button>
									)}
									<Button
										size="sm"
										variant="ghost"
										onClick={toggleMute}
										className="text-white hover:bg-white/10"
									>
										{isMuted ? (
											<VolumeX className="w-4 h-4" />
										) : (
											<Volume2 className="w-4 h-4" />
										)}
									</Button>
									<Button
										size="sm"
										variant="ghost"
										onClick={handleDownload}
										className="text-green-400 hover:bg-green-500/10 hover:text-green-300"
									>
										<Download className="w-4 h-4 mr-1" />
										Download
									</Button>
									<Button
										size="sm"
										variant="ghost"
										onClick={onClose}
										className="text-white hover:bg-white/10"
									>
										<X className="w-4 h-4" />
									</Button>
								</div>
							</div>
							{location && (
								<div className="mt-2 text-sm text-gray-300">
									{location}
								</div>
							)}
						</div>

						{/* Video Player or Unavailable Overlay */}
						<div className="relative w-full h-full flex items-center justify-center bg-black">
							{offlineMode ? (
								<div className="absolute inset-0 flex items-center justify-center bg-black/95">
									<div className="flex flex-col items-center justify-center space-y-4">
										<div className="relative">
											<div className="absolute inset-0 bg-red-500/20 blur-xl rounded-full animate-pulse" />
											<WifiOff className="w-16 h-16 text-zinc-600 relative z-10" />
										</div>
										<div className="text-center space-y-2">
											<p className="text-zinc-500 font-mono text-xl tracking-[0.2em] font-bold">SYSTEM OFFLINE</p>
											<p className="text-zinc-700 text-sm font-medium uppercase tracking-widest">Live Feed Disconnected</p>
										</div>
									</div>
								</div>
							) : !(videoError || !videoUrl) ? (
								<>
									<video
										ref={videoRef}
										src={videoUrl}
										controls
										className="w-full h-full object-contain"
										onPlay={() => setIsPlaying(true)}
										onPause={() => setIsPlaying(false)}
										onError={() => setVideoError(true)}
										muted={isMuted}
									>
										Your browser does not support the video tag.
									</video>
									{/* Hover Overlay with Play/Pause */}
									<div
										className="absolute inset-0 bg-black/0 hover:bg-black/30 transition-colors flex items-center justify-center cursor-pointer group"
										onClick={togglePlayPause}
									>
										<div className="opacity-0 group-hover:opacity-100 transition-opacity">
											{isPlaying ? (
												<div className="w-16 h-16 rounded-full bg-white/20 backdrop-blur flex items-center justify-center">
													<Pause className="w-8 h-8 text-white" />
												</div>
											) : (
												<div className="w-16 h-16 rounded-full bg-white/20 backdrop-blur flex items-center justify-center">
													<Play className="w-8 h-8 text-white ml-1" />
												</div>
											)}
										</div>
									</div>
								</>
							) : (
								<div className="absolute inset-0 z-[99999] flex items-center justify-center bg-gradient-to-br from-blue-900 via-cyan-900 to-blue-700" style={{ backgroundColor: '#0a2540' }}>
									<div className="text-center px-8 py-6 rounded-xl shadow-2xl bg-blue-950/80 border border-cyan-700/40">
										<div className="flex justify-center mb-4">
											<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-16 h-16 text-cyan-400">
												<path strokeLinecap="round" strokeLinejoin="round" d="M15.75 9V5.25A2.25 2.25 0 0013.5 3h-3A2.25 2.25 0 008.25 5.25V9m7.5 0v10.5A2.25 2.25 0 0113.5 21h-3a2.25 2.25 0 01-2.25-2.25V9m7.5 0h-7.5" />
											</svg>
										</div>
										<p className="text-cyan-200 text-lg font-bold">Video not available</p>
										<p className="text-cyan-400 text-sm mt-2">The video stream could not be loaded</p>
									</div>
								</div>
							)}
						</div>

						{/* Footer with camera info and custom controls */}
						<div className="absolute bottom-0 left-0 right-0 z-10 bg-gradient-to-t from-black/90 via-black/70 to-transparent backdrop-blur-md border-t border-cyan-500/20 p-4">
							<div className="flex flex-col gap-2">
								{controls && (
									<div>{controls}</div>
								)}
								<div className="flex items-center justify-between text-sm">
									<span className="text-gray-400">
										Camera: <span className="text-cyan-400 font-mono">{cameraId}</span>
									</span>
									<span className="text-gray-400 font-mono">
										{new Date().toLocaleTimeString('en-US', {
											hour: '2-digit',
											minute: '2-digit',
											second: '2-digit',
											hour12: false,
										})}
									</span>
								</div>
							</div>
						</div>
					</motion.div>
				</motion.div>
			)}
		</AnimatePresence>
	);
}
