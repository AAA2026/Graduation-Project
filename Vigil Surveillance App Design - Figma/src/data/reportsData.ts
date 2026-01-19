export interface Report {
    id: string;
    title: string;
    type: 'violence' | 'crash' | 'analytics' | 'cameras' | 'performance' | 'custom' | 'incidents'; // Union of all types
    status: 'completed' | 'processing' | 'failed';
    generatedDate: string;
    generatedBy: string;
    size: string;
    format: string;
    incidentsCount?: number;
    camerasCount?: number;
}

export const SHARED_REPORTS: Report[] = [
    {
        id: 'RPT-2024-001',
        title: 'December Incident Summary',
        type: 'incidents',
        status: 'completed',
        generatedDate: '2024-12-28 14:30',
        generatedBy: 'System',
        size: '2.4 MB',
        format: 'PDF',
        incidentsCount: 45,
        camerasCount: 12
    },
    {
        id: 'RPT-2025-001',
        title: 'Monthly Violence Detection Report',
        type: 'violence',
        status: 'completed',
        generatedDate: '2026-01-09 09:00',
        generatedBy: 'Admin',
        size: '2.4 MB',
        format: 'PDF',
        incidentsCount: 45,
        camerasCount: 12
    },
    {
        id: 'RPT-2025-002',
        title: 'Highway Crash Analysis Q4 2025',
        type: 'crash',
        status: 'completed',
        generatedDate: '2026-01-08 15:20',
        generatedBy: 'Officer Martinez',
        size: '1.8 MB',
        format: 'PDF',
        incidentsCount: 23,
        camerasCount: 8
    },
    {
        id: 'RPT-2025-003',
        title: 'Downtown Zone Security Summary',
        type: 'custom',
        status: 'completed',
        generatedDate: '2026-01-07 11:45',
        generatedBy: 'Admin',
        size: '3.1 MB',
        format: 'PDF',
        incidentsCount: 67,
        camerasCount: 15
    },
    {
        id: 'RPT-2025-004',
        title: 'Weekly Incident Overview',
        type: 'violence',
        status: 'completed',
        generatedDate: '2026-01-06 08:30',
        generatedBy: 'Officer Chen',
        size: '1.2 MB',
        format: 'PDF',
        incidentsCount: 12,
        camerasCount: 20
    },
    {
        id: 'RPT-2025-005',
        title: 'Camera Performance Analysis',
        type: 'cameras',
        status: 'processing',
        generatedDate: '2026-01-05 10:15',
        generatedBy: 'Admin',
        size: '-',
        format: 'PDF',
        incidentsCount: 0,
        camerasCount: 152
    },
    {
        id: 'RPT-2024-002',
        title: 'Weekly Analytics',
        type: 'analytics',
        status: 'completed',
        generatedDate: '2024-12-28 10:15',
        generatedBy: 'System',
        size: '1.8 MB',
        format: 'PDF'
    },
    {
        id: 'RPT-2024-004',
        title: 'Q4 Performance Review',
        type: 'performance',
        status: 'completed',
        generatedDate: '2024-12-27 16:45',
        generatedBy: 'HR',
        size: '3.1 MB',
        format: 'PDF'
    }
].sort((a, b) => new Date(b.generatedDate).getTime() - new Date(a.generatedDate).getTime());
