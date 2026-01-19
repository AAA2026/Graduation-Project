import { Card, CardHeader, CardTitle, CardContent } from './ui/card';
import { Tabs, TabsList, TabsTrigger, TabsContent } from './ui/tabs';
import { CalendarDays, Users, CheckCircle, Clock, PlusCircle } from 'lucide-react';
import { useState, useEffect } from 'react';
import { Button } from './ui/button';


interface DemoRequest {
  id: string;
  fullName: string;
  email: string;
  phone: string;
  organization: string;
  role: string;
  cameras: string;
  message: string;
  status: 'pending' | 'scheduled' | 'completed' | string;
  created_at: string;
  created_at_human: string;
  updated_at?: string;
}

export function DemoRequestsTab() {
    // Update booking status
    const updateStatus = async (id: string, status: string) => {
      try {
        setLoading(true);
        const res = await fetch(`/api/demo-bookings/${id}/status`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ status }),
        });
        if (!res.ok) throw new Error('Failed to update status');
        // Refresh bookings
        const updated = await fetch('/api/demo-bookings').then(r => r.json());
        setRequests(updated);
        setLoading(false);
      } catch (e) {
        setError('Failed to update status');
        setLoading(false);
      }
    };
  const [tab, setTab] = useState('all');
  const [requests, setRequests] = useState<DemoRequest[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch real demo bookings from backend
  useEffect(() => {
    setLoading(true);
    fetch('/api/demo-bookings')
      .then((res) => res.json())
      .then((data) => {
        setRequests(data);
        setLoading(false);
      })
      .catch((err) => {
        setError('Failed to load demo requests');
        setLoading(false);
      });
  }, []);

  const filtered = (status: string | 'all') =>
    status === 'all' ? requests : requests.filter(r => r.status === status);

  return (
    <Card className="bg-card/80 backdrop-blur border-border">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <CalendarDays className="w-7 h-7 text-blue-600 dark:text-blue-400 relative" style={{ top: '-2px' }} />
            <h1 className="text-2xl font-bold tracking-tight">Demo Requests & Scheduling</h1>
          </div>
          <div className="flex items-center gap-2">
            <Button size="sm" variant="outline" title="New Request">
              <PlusCircle className="w-4 h-4 mr-1" /> New Request
            </Button>
            <Button size="sm" variant="outline" title="Export CSV">
              <CheckCircle className="w-4 h-4 mr-1" /> Export CSV
            </Button>
            <Button size="sm" variant="outline" title="Refresh">
              <Clock className="w-4 h-4 mr-1" /> Refresh
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <Tabs value={tab} onValueChange={setTab}>
          <TabsList className="mb-4">
            <TabsTrigger value="all">All</TabsTrigger>
            <TabsTrigger value="pending">Pending</TabsTrigger>
            <TabsTrigger value="scheduled">Scheduled</TabsTrigger>
            <TabsTrigger value="completed">Completed</TabsTrigger>
          </TabsList>
          {['all', 'pending', 'scheduled', 'completed'].map((status) => (
            <TabsContent key={status} value={status}>
              {loading ? (
                <div className="text-muted-foreground text-sm">Loading...</div>
              ) : error ? (
                <div className="text-red-500 text-sm">{error}</div>
              ) : (
                <div className="space-y-3">
                  {filtered(status).length === 0 && (
                    <div className="text-muted-foreground text-sm">No requests.</div>
                  )}
                  {filtered(status).map((req) => (
                    <div key={req.id} className="flex items-center justify-between p-4 rounded-lg bg-muted/50 border border-border">
                      <div className="flex items-center gap-4">
                        <Users className="w-5 h-5 text-blue-600 dark:text-blue-400" />
                        <div>
                          <div className="font-semibold text-foreground">{req.fullName}</div>
                          <div className="text-xs text-muted-foreground">{req.organization}</div>
                          <div className="text-xs text-muted-foreground">{req.email}</div>
                          <div className="text-xs text-muted-foreground">{req.phone}</div>
                          <div className="text-xs text-muted-foreground">Role: {req.role}</div>
                          <div className="text-xs text-muted-foreground">Cameras: {req.cameras}</div>
                          <div className="text-xs text-muted-foreground">Requested: {req.created_at_human}</div>
                          {req.message && (
                            <div className="text-xs text-muted-foreground">Message: {req.message}</div>
                          )}
                          {req.updated_at && (
                            <div className="text-xs text-blue-500">Updated: {req.updated_at}</div>
                          )}
                        </div>
                      </div>
                      <div className="flex flex-col items-end gap-2">
                        {req.status === 'pending' && <>
                          <Clock className="w-4 h-4 text-amber-500" />
                          <Button size="sm" className="mt-1 bg-blue-600 text-white" onClick={() => updateStatus(req.id, 'scheduled')}>Schedule</Button>
                        </>}
                        {req.status === 'scheduled' && <>
                          <CheckCircle className="w-4 h-4 text-blue-500" />
                          <Button size="sm" className="mt-1 bg-green-600 text-white" onClick={() => updateStatus(req.id, 'completed')}>Mark Completed</Button>
                        </>}
                        {req.status === 'completed' && <CheckCircle className="w-4 h-4 text-green-500" />}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </TabsContent>
          ))}
        </Tabs>
      </CardContent>
    </Card>
  );
}
