import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Badge } from './ui/badge';
import { ScrollArea } from './ui/scroll-area';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from './ui/table';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from './ui/select';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from './ui/dialog';
import {
  Camera,
  Plus,
  Search,
  Filter,
  Radio,
  Settings,
  Eye,
  AlertCircle,
  MapPin,
  RefreshCw,
} from 'lucide-react';
import { toast } from 'sonner';
import { useTheme } from './ThemeProvider';

export function CameraManagement() {
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');
  const [zoneFilter, setZoneFilter] = useState('all');
  const { theme } = useTheme();

  const cameras = [
    {
      id: 'CAM-042',
      name: 'Main Street Camera 1',
      location: 'Main St & 5th Ave',
      zone: 'Downtown',
      status: 'online',
      uptime: '99.8%',
      lastSeen: '2 min ago',
      incidents: 12,
      resolution: '4K',
      fps: 30,
    },
    {
      id: 'CAM-128',
      name: 'Highway Monitor 23',
      location: 'Highway 101, Exit 23',
      zone: 'Highway',
      status: 'online',
      uptime: '99.5%',
      lastSeen: '1 min ago',
      incidents: 8,
      resolution: '1080p',
      fps: 60,
    },
    {
      id: 'CAM-089',
      name: 'Park Plaza Surveillance',
      location: 'Park Plaza, North Wing',
      zone: 'Downtown',
      status: 'online',
      uptime: '98.2%',
      lastSeen: '3 min ago',
      incidents: 5,
      resolution: '4K',
      fps: 30,
    },
    {
      id: 'CAM-156',
      name: 'Downtown Bridge Monitor',
      location: 'Downtown Bridge',
      zone: 'Downtown',
      status: 'maintenance',
      uptime: '95.1%',
      lastSeen: '45 min ago',
      incidents: 15,
      resolution: '1080p',
      fps: 30,
    },
    {
      id: 'CAM-203',
      name: 'Shopping District Cam',
      location: 'Shopping District, Block C',
      zone: 'Residential',
      status: 'online',
      uptime: '99.9%',
      lastSeen: '1 min ago',
      incidents: 3,
      resolution: '4K',
      fps: 60,
    },
    {
      id: 'CAM-074',
      name: 'Riverside Parkway',
      location: 'Riverside Parkway',
      zone: 'Residential',
      status: 'offline',
      uptime: '92.3%',
      lastSeen: '2 hours ago',
      incidents: 7,
      resolution: '1080p',
      fps: 30,
    },
    {
      id: 'CAM-112',
      name: 'Industrial Park A',
      location: 'Industrial Park A',
      zone: 'Industrial',
      status: 'online',
      uptime: '99.2%',
      lastSeen: '5 min ago',
      incidents: 2,
      resolution: '1080p',
      fps: 30,
    },
    {
      id: 'CAM-198',
      name: 'North Terminal',
      location: 'North Terminal',
      zone: 'Highway',
      status: 'online',
      uptime: '99.6%',
      lastSeen: '2 min ago',
      incidents: 11,
      resolution: '4K',
      fps: 60,
    },
  ];

  const filteredCameras = cameras.filter((camera) => {
    if (statusFilter !== 'all' && camera.status !== statusFilter) return false;
    if (zoneFilter !== 'all' && camera.zone !== zoneFilter) return false;
    if (
      searchQuery &&
      !camera.name.toLowerCase().includes(searchQuery.toLowerCase()) &&
      !camera.id.toLowerCase().includes(searchQuery.toLowerCase()) &&
      !camera.location.toLowerCase().includes(searchQuery.toLowerCase())
    ) {
      return false;
    }
    return true;
  });

  const handleRestart = (cameraId: string) => {
    toast.success(`Restarting camera ${cameraId}`);
  };

  const handleConfigure = (cameraId: string) => {
    toast.info(`Opening settings for ${cameraId}`);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
        <div className="flex items-center gap-2">
          <Camera className="w-8 h-8 text-blue-600 dark:text-blue-400 relative" style={{ top: '-2px' }} />
          <h2 className="text-2xl font-bold text-foreground mb-1">Camera Management</h2>
        </div>
        {/* Subtext removed as requested */}
        <Dialog>
          <DialogTrigger asChild>
            <Button className="bg-blue-600 hover:bg-blue-700 text-gray-900 dark:text-white">
              <Plus className="w-4 h-4 mr-2" />
              Add Camera
            </Button>
          </DialogTrigger>
          <DialogContent className="bg-card border-border text-foreground">
            <DialogHeader>
              <DialogTitle>Add New Camera</DialogTitle>
              <DialogDescription className="text-muted-foreground">
                Register a new camera to the surveillance network
              </DialogDescription>
            </DialogHeader>
            <div className="space-y-4 mt-4">
              <div className="space-y-2">
                <label className="text-sm text-foreground">Camera ID</label>
                <Input
                  placeholder="CAM-XXX"
                  className="bg-background border-border text-foreground"
                />
              </div>
              <div className="space-y-2">
                <label className="text-sm text-foreground">Camera Name</label>
                <Input
                  placeholder="Enter camera name"
                  className="bg-background border-border text-foreground"
                />
              </div>
              <div className="space-y-2">
                <label className="text-sm text-foreground">Location</label>
                <Input
                  placeholder="Enter location"
                  className="bg-background border-border text-foreground"
                />
              </div>
              <div className="space-y-2">
                <label className="text-sm text-foreground">Zone</label>
                <Select>
                  <SelectTrigger className="bg-background border-border text-foreground">
                    <SelectValue placeholder="Select zone" />
                  </SelectTrigger>
                  <SelectContent className="bg-card border-border text-foreground">
                    <SelectItem value="downtown">Downtown</SelectItem>
                    <SelectItem value="highway">Highway</SelectItem>
                    <SelectItem value="residential">Residential</SelectItem>
                    <SelectItem value="industrial">Industrial</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <Button className="w-full bg-blue-600 hover:bg-blue-700 text-gray-900 dark:text-white">
                Add Camera
              </Button>
            </div>
          </DialogContent>
        </Dialog>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="bg-gradient-to-br from-green-500/20 to-green-600/20 border-green-500/30">
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-green-500/20 flex items-center justify-center">
                <Radio className="w-5 h-5 text-green-600 dark:text-green-400" />
              </div>
              <div>
                <div className="text-2xl text-foreground">245</div>
                <div className="text-sm text-muted-foreground">Online</div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-br from-red-500/20 to-red-600/20 border-red-500/30">
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-red-500/20 flex items-center justify-center">
                <AlertCircle className="w-5 h-5 text-red-600 dark:text-red-400" />
              </div>
              <div>
                <div className="text-2xl text-foreground">2</div>
                <div className="text-sm text-muted-foreground">Offline</div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-br from-amber-500/20 to-amber-600/20 border-amber-500/30">
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-amber-500/20 flex items-center justify-center">
                <Settings className="w-5 h-5 text-amber-600 dark:text-amber-400" />
              </div>
              <div>
                <div className="text-2xl text-foreground">1</div>
                <div className="text-sm text-muted-foreground">Maintenance</div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-br from-blue-500/20 to-blue-600/20 border-blue-500/30">
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-blue-500/20 flex items-center justify-center">
                <Camera className="w-5 h-5 text-blue-600 dark:text-blue-400" />
              </div>
              <div>
                <div className="text-2xl text-foreground">248</div>
                <div className="text-sm text-muted-foreground">Total</div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Filters */}
      <Card className="bg-card/80 backdrop-blur border-border">
        <CardContent className="p-4">
          <div className="flex flex-col gap-3 md:flex-row md:items-center">
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
              <Input
                placeholder="Search cameras..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10 bg-background border-border text-foreground placeholder:text-muted-foreground"
              />
            </div>
            <Select value={statusFilter} onValueChange={setStatusFilter}>
              <SelectTrigger className="w-full md:w-40 bg-background border-border text-foreground">
                <SelectValue placeholder="Status" />
              </SelectTrigger>
              <SelectContent className="bg-card border-border text-foreground">
                <SelectItem value="all">All Status</SelectItem>
                <SelectItem value="online">Online</SelectItem>
                <SelectItem value="offline">Offline</SelectItem>
                <SelectItem value="maintenance">Maintenance</SelectItem>
              </SelectContent>
            </Select>
            <Select value={zoneFilter} onValueChange={setZoneFilter}>
              <SelectTrigger className="w-full md:w-40 bg-background border-border text-foreground">
                <SelectValue placeholder="Zone" />
              </SelectTrigger>
              <SelectContent className="bg-card border-border text-foreground">
                <SelectItem value="all">All Zones</SelectItem>
                <SelectItem value="Downtown">Downtown</SelectItem>
                <SelectItem value="Highway">Highway</SelectItem>
                <SelectItem value="Residential">Residential</SelectItem>
                <SelectItem value="Industrial">Industrial</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      {/* Camera Table */}
      <Card className="bg-card/80 backdrop-blur border-border">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-card-foreground">
            <Camera className="w-5 h-5 text-blue-600 dark:text-blue-400" />
            Camera List ({filteredCameras.length})
          </CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          <ScrollArea className="h-[600px]">
            <Table>
              <TableHeader>
                <TableRow className="border-border hover:bg-transparent">
                  <TableHead className="text-muted-foreground">Camera ID</TableHead>
                  <TableHead className="text-muted-foreground">Name</TableHead>
                  <TableHead className="text-muted-foreground">Location</TableHead>
                  <TableHead className="text-muted-foreground">Zone</TableHead>
                  <TableHead className="text-muted-foreground">Status</TableHead>
                  <TableHead className="text-muted-foreground">Uptime</TableHead>
                  <TableHead className="text-muted-foreground">Incidents</TableHead>
                  <TableHead className="text-muted-foreground">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {filteredCameras.map((camera) => (
                  <TableRow key={camera.id} className="border-border hover:bg-muted/50">
                    <TableCell className="text-foreground">{camera.id}</TableCell>
                    <TableCell className="text-foreground">{camera.name}</TableCell>
                    <TableCell className="text-muted-foreground">
                      <div className="flex items-center gap-2">
                        <MapPin className="w-3 h-3" />
                        {camera.location}
                      </div>
                    </TableCell>
                    <TableCell>
                      <Badge variant="outline" className="border-border text-foreground">
                        {camera.zone}
                      </Badge>
                    </TableCell>
                    <TableCell>
                      <Badge
                        className={
                          camera.status === 'online'
                            ? 'bg-green-100 text-green-800 border-green-200 dark:bg-green-500/20 dark:text-green-400 dark:border-green-500/30'
                            : camera.status === 'offline'
                            ? 'bg-red-100 text-red-800 border-red-200 dark:bg-red-500/20 dark:text-red-400 dark:border-red-500/30'
                            : 'bg-amber-100 text-amber-800 border-amber-200 dark:bg-amber-500/20 dark:text-amber-400 dark:border-amber-500/30'
                        }
                      >
                        <Radio className="w-3 h-3 mr-1" />
                        {camera.status}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-muted-foreground">{camera.uptime}</TableCell>
                    <TableCell className="text-foreground">{camera.incidents}</TableCell>
                    <TableCell>
                      <div className="flex items-center gap-2">
                        <Button
                          size="icon"
                          variant="ghost"
                          className="w-8 h-8 text-muted-foreground hover:text-foreground"
                          onClick={() => handleConfigure(camera.id)}
                        >
                          <Settings className="w-4 h-4" />
                        </Button>
                        <Button
                          size="icon"
                          variant="ghost"
                          className="w-8 h-8 text-muted-foreground hover:text-foreground"
                          onClick={() => handleRestart(camera.id)}
                        >
                          <RefreshCw className="w-4 h-4" />
                        </Button>
                        <Button
                          size="icon"
                          variant="ghost"
                          className="w-8 h-8 text-blue-600 hover:text-blue-700 dark:text-blue-400 dark:hover:text-blue-300"
                        >
                          <Eye className="w-4 h-4" />
                        </Button>
                      </div>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </ScrollArea>
        </CardContent>
      </Card>
    </div>
  );
}