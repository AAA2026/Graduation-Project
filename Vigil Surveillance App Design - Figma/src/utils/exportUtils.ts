import jsPDF from 'jspdf';
import autoTable from 'jspdf-autotable';
import html2canvas from 'html2canvas';

// Types
interface Incident {
  id: string;
  timestamp: string;
  cameraId: string;
  cameraName: string;
  type: string;
  confidence: number;
  location: string;
  status: string;
  assignedOfficer?: string;
  responseTime?: number;
  aiModel?: string;
  videoUrl?: string;
}

interface AnalyticsData {
  totalIncidents: number;
  avgResponseTime: number;
  activeAlerts: number;
  avgConfidence: number;
  incidentsByType?: Record<string, number>;
  incidentsByZone?: Record<string, number>;
}

interface CameraData {
  id: string;
  name: string;
  location: string;
  status: string;
  zone: string;
  model?: string;
  resolution?: string;
  fps?: number;
}

// ======================
// PDF EXPORT FUNCTIONS
// ======================

/**
 * Export incidents to PDF report
 */
export const exportIncidentsToPDF = (incidents: Incident[]) => {
  const doc = new jsPDF();
  
  // Header
  doc.setFontSize(20);
  doc.text('VIGIL Incident Report', 14, 22);
  
  doc.setFontSize(10);
  doc.setTextColor(100);
  doc.text(`Generated: ${new Date().toLocaleString()}`, 14, 30);
  doc.text(`Total Incidents: ${incidents.length}`, 14, 36);
  
  // Table data
  const tableData = incidents.map(inc => [
    inc.id,
    new Date(inc.timestamp).toLocaleString(),
    inc.cameraName,
    inc.type,
    `${(inc.confidence * 100).toFixed(1)}%`,
    inc.location,
    inc.status,
    inc.assignedOfficer || 'Unassigned',
    inc.responseTime ? `${inc.responseTime}s` : 'Pending',
  ]);

  // Add table
  autoTable(doc, {
    head: [['ID', 'Timestamp', 'Camera', 'Type', 'Confidence', 'Location', 'Status', 'Officer', 'Response Time']],
    body: tableData,
    startY: 45,
    styles: { fontSize: 8, cellPadding: 2 },
    headStyles: { fillColor: [6, 182, 212], textColor: 255 },
    alternateRowStyles: { fillColor: [245, 245, 245] },
    columnStyles: {
      0: { cellWidth: 25 },
      1: { cellWidth: 35 },
      2: { cellWidth: 25 },
      3: { cellWidth: 20 },
      4: { cellWidth: 18 },
      5: { cellWidth: 25 },
      6: { cellWidth: 18 },
      7: { cellWidth: 22 },
      8: { cellWidth: 20 },
    },
  });

  // Footer
  const pageCount = (doc as any).internal.getNumberOfPages();
  doc.setFontSize(8);
  doc.setTextColor(150);
  for (let i = 1; i <= pageCount; i++) {
    doc.setPage(i);
    doc.text(
      `Page ${i} of ${pageCount}`,
      doc.internal.pageSize.getWidth() / 2,
      doc.internal.pageSize.getHeight() - 10,
      { align: 'center' }
    );
  }

  // Save
  const filename = `vigil_incident_report_${Date.now()}.pdf`;
  doc.save(filename);
  
  return filename;
};

/**
 * Export analytics summary to PDF
 */
export const exportAnalyticsToPDF = (data: AnalyticsData) => {
  const doc = new jsPDF();
  
  // Header
  doc.setFontSize(20);
  doc.text('VIGIL Analytics Report', 14, 22);
  
  doc.setFontSize(10);
  doc.setTextColor(100);
  doc.text(`Generated: ${new Date().toLocaleString()}`, 14, 30);
  
  // Summary Stats
  doc.setFontSize(12);
  doc.setTextColor(0);
  doc.text('Summary Statistics', 14, 45);
  
  const summaryData = [
    ['Total Incidents', data.totalIncidents.toString()],
    ['Average Response Time', `${data.avgResponseTime.toFixed(1)}s`],
    ['Active Alerts', data.activeAlerts.toString()],
    ['Average Confidence', `${(data.avgConfidence * 100).toFixed(1)}%`],
  ];
  
  autoTable(doc, {
    body: summaryData,
    startY: 50,
    theme: 'striped',
    headStyles: { fillColor: [6, 182, 212] },
  });

  // Incidents by Type
  if (data.incidentsByType) {
    doc.setFontSize(12);
    doc.text('Incidents by Type', 14, (doc as any).lastAutoTable.finalY + 15);
    
    const typeData = Object.entries(data.incidentsByType).map(([type, count]) => [type, count.toString()]);
    
    autoTable(doc, {
      head: [['Incident Type', 'Count']],
      body: typeData,
      startY: (doc as any).lastAutoTable.finalY + 20,
      theme: 'striped',
      headStyles: { fillColor: [6, 182, 212] },
    });
  }

  // Incidents by Zone
  if (data.incidentsByZone) {
    doc.setFontSize(12);
    doc.text('Incidents by Zone', 14, (doc as any).lastAutoTable.finalY + 15);
    
    const zoneData = Object.entries(data.incidentsByZone).map(([zone, count]) => [zone, count.toString()]);
    
    autoTable(doc, {
      head: [['Zone', 'Count']],
      body: zoneData,
      startY: (doc as any).lastAutoTable.finalY + 20,
      theme: 'striped',
      headStyles: { fillColor: [6, 182, 212] },
    });
  }

  const filename = `vigil_analytics_report_${Date.now()}.pdf`;
  doc.save(filename);
  
  return filename;
};

// ======================
// CSV EXPORT FUNCTIONS
// ======================

/**
 * Export incidents to CSV
 */
export const exportIncidentsToCSV = (incidents: Incident[]) => {
  const headers = [
    'ID',
    'Timestamp',
    'Camera ID',
    'Camera Name',
    'Type',
    'Confidence (%)',
    'Location',
    'Status',
    'Assigned Officer',
    'Response Time (s)',
    'AI Model',
    'Video URL'
  ];

  const rows = incidents.map(inc => [
    inc.id,
    new Date(inc.timestamp).toISOString(),
    inc.cameraId,
    inc.cameraName,
    inc.type,
    (inc.confidence * 100).toFixed(2),
    inc.location,
    inc.status,
    inc.assignedOfficer || '',
    inc.responseTime?.toString() || '',
    inc.aiModel || '',
    inc.videoUrl || ''
  ]);

  const csvContent = [
    headers.join(','),
    ...rows.map(row => row.map(cell => `"${cell}"`).join(','))
  ].join('\n');

  downloadCSV(csvContent, `vigil_incidents_${Date.now()}.csv`);
};

/**
 * Export analytics data to CSV
 */
export const exportAnalyticsToCSV = (data: AnalyticsData) => {
  const rows = [
    ['Metric', 'Value'],
    ['Total Incidents', data.totalIncidents.toString()],
    ['Average Response Time (s)', data.avgResponseTime.toFixed(2)],
    ['Active Alerts', data.activeAlerts.toString()],
    ['Average Confidence (%)', (data.avgConfidence * 100).toFixed(2)],
  ];

  if (data.incidentsByType) {
    rows.push(['', '']);
    rows.push(['Incidents by Type', '']);
    Object.entries(data.incidentsByType).forEach(([type, count]) => {
      rows.push([type, count.toString()]);
    });
  }

  if (data.incidentsByZone) {
    rows.push(['', '']);
    rows.push(['Incidents by Zone', '']);
    Object.entries(data.incidentsByZone).forEach(([zone, count]) => {
      rows.push([zone, count.toString()]);
    });
  }

  const csvContent = rows.map(row => row.map(cell => `"${cell}"`).join(',')).join('\n');
  downloadCSV(csvContent, `vigil_analytics_${Date.now()}.csv`);
};

/**
 * Export camera data to CSV
 */
export const exportCameraDataToCSV = (cameras: CameraData[]) => {
  const headers = ['ID', 'Name', 'Location', 'Status', 'Zone', 'Model', 'Resolution', 'FPS'];

  const rows = cameras.map(cam => [
    cam.id,
    cam.name,
    cam.location,
    cam.status,
    cam.zone,
    cam.model || '',
    cam.resolution || '',
    cam.fps?.toString() || ''
  ]);

  const csvContent = [
    headers.join(','),
    ...rows.map(row => row.map(cell => `"${cell}"`).join(','))
  ].join('\n');

  downloadCSV(csvContent, `vigil_cameras_${Date.now()}.csv`);
};

// ======================
// SCREENSHOT FUNCTIONS
// ======================

/**
 * Capture screenshot of an element
 */
export const captureScreenshot = async (elementId: string, filename?: string) => {
  const element = document.getElementById(elementId);
  
  if (!element) {
    console.error(`Element with id "${elementId}" not found`);
    return;
  }

  try {
    const canvas = await html2canvas(element, {
      backgroundColor: '#0a0e1a',
      scale: 2,
      logging: false,
    });

    const link = document.createElement('a');
    link.download = filename || `vigil_screenshot_${Date.now()}.png`;
    link.href = canvas.toDataURL('image/png');
    link.click();
  } catch (error) {
    console.error('Screenshot capture failed:', error);
  }
};

// ======================
// VIDEO DOWNLOAD FUNCTIONS
// ======================

/**
 * Download video clip from URL
 */
export const downloadVideoClip = async (
  videoUrl: string,
  cameraId: string,
  timestamp: string
): Promise<void> => {
  try {
    // Since we're in frontend-only mode, simulate the download
    // In production, this would fetch from the actual backend
    
    // Check if this is a mock URL (localhost backend that doesn't exist yet)
    const isMockUrl = videoUrl.includes('127.0.0.1:5000') || videoUrl.includes('localhost:5000');
    
    if (isMockUrl) {
      // Simulate a download with a placeholder video
      // Create a small mock video blob (just a placeholder)
      const mockVideoData = new Blob(
        [new Uint8Array([0x00, 0x00, 0x00, 0x20, 0x66, 0x74, 0x79, 0x70])],
        { type: 'video/mp4' }
      );
      
      // Create filename with timestamp
      const timestampFormatted = new Date(timestamp).getTime();
      const filename = `${cameraId}_${timestampFormatted}_mock.mp4`;
      
      // Create download link
      const url = URL.createObjectURL(mockVideoData);
      const link = document.createElement('a');
      link.href = url;
      link.download = filename;
      link.style.display = 'none';
      
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      
      // Cleanup
      setTimeout(() => URL.revokeObjectURL(url), 100);
      
      // Note: In production, replace this with actual backend fetch
      console.log(`Mock download: ${filename} (Backend integration pending)`);
    } else {
      // For real URLs, attempt to fetch
      const response = await fetch(videoUrl);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch video: ${response.statusText}`);
      }

      // Create a blob from the response
      const blob = await response.blob();

      // Create filename with timestamp
      const timestampFormatted = new Date(timestamp).getTime();
      const filename = `${cameraId}_${timestampFormatted}.mp4`;

      // Create download link
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = filename;
      link.style.display = 'none';
      
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);

      // Cleanup
      setTimeout(() => URL.revokeObjectURL(url), 100);
    }
  } catch (error) {
    console.error('Video download failed:', error);
    throw error;
  }
};

/**
 * Capture all camera feeds
 */
export const captureAllCameraFeeds = async () => {
  const cameraGrid = document.querySelector('[data-camera-grid]');
  
  if (!cameraGrid) {
    console.error('Camera grid not found');
    return;
  }

  try {
    const canvas = await html2canvas(cameraGrid as HTMLElement, {
      backgroundColor: '#000',
      scale: 2,
      logging: false,
    });

    const link = document.createElement('a');
    link.download = `vigil_camera_grid_${Date.now()}.png`;
    link.href = canvas.toDataURL('image/png');
    link.click();
  } catch (error) {
    console.error('Camera grid screenshot failed:', error);
  }
};

// ======================
// HELPER FUNCTIONS
// ======================

/**
 * Download CSV file
 */
const downloadCSV = (content: string, filename: string) => {
  const blob = new Blob([content], { type: 'text/csv;charset=utf-8;' });
  const link = document.createElement('a');
  
  if (navigator.msSaveBlob) {
    // IE 10+
    navigator.msSaveBlob(blob, filename);
  } else {
    link.href = URL.createObjectURL(blob);
    link.download = filename;
    link.style.display = 'none';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }
};

/**
 * Format date for exports
 */
export const formatDateForExport = (date: Date | string): string => {
  const d = typeof date === 'string' ? new Date(date) : date;
  return d.toISOString().split('T')[0]; // YYYY-MM-DD
};

/**
 * Format time for exports
 */
export const formatTimeForExport = (date: Date | string): string => {
  const d = typeof date === 'string' ? new Date(date) : date;
  return d.toTimeString().split(' ')[0]; // HH:MM:SS
};