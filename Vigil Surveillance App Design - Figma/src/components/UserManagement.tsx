import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Badge } from './ui/badge';
import { Avatar, AvatarFallback } from './ui/avatar';
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
import { Label } from './ui/label';
import {
  Users,
  Plus,
  Search,
  Shield,
  Eye,
  Lock,
  MoreVertical,
  Edit,
  Trash2,
  Mail,
  Phone,
  ChevronRight,
  User,
} from 'lucide-react';
import { toast } from 'sonner@2.0.3';
import { useTheme } from './ThemeProvider';
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetTrigger } from './ui/sheet';

export function UserManagement() {
  const [searchQuery, setSearchQuery] = useState('');
  const [roleFilter, setRoleFilter] = useState('all');
  const [selectedUser, setSelectedUser] = useState<number | null>(null);
  const [isMobileView, setIsMobileView] = useState(false);
  const { theme } = useTheme();

  // Check for mobile view on mount and resize
  useEffect(() => {
    const checkMobile = () => {
      setIsMobileView(window.innerWidth < 768);
    };
    
    checkMobile();
    window.addEventListener('resize', checkMobile);
    return () => window.removeEventListener('resize', checkMobile);
  }, []);

  const users = [
    {
      id: 1,
      name: 'Sarah Anderson',
      email: 'sarah.anderson@vigil.com',
      role: 'Admin',
      status: 'active',
      lastLogin: '2 hours ago',
      phone: '+1 (555) 123-4567',
      incidents: 142,
    },
    {
      id: 2,
      name: 'Michael Chen',
      email: 'michael.chen@vigil.com',
      role: 'Officer',
      status: 'active',
      lastLogin: '5 minutes ago',
      phone: '+1 (555) 234-5678',
      incidents: 89,
    },
    {
      id: 3,
      name: 'Jessica Martinez',
      email: 'jessica.martinez@vigil.com',
      role: 'Officer',
      status: 'active',
      lastLogin: '1 hour ago',
      phone: '+1 (555) 345-6789',
      incidents: 67,
    },
    {
      id: 4,
      name: 'David Thompson',
      email: 'david.thompson@vigil.com',
      role: 'Security',
      status: 'active',
      lastLogin: '10 minutes ago',
      phone: '+1 (555) 456-7890',
      incidents: 45,
    },
    {
      id: 5,
      name: 'Emily Rodriguez',
      email: 'emily.rodriguez@vigil.com',
      role: 'Admin',
      status: 'active',
      lastLogin: '3 hours ago',
      phone: '+1 (555) 567-8901',
      incidents: 156,
    },
    {
      id: 6,
      name: 'James Wilson',
      email: 'james.wilson@vigil.com',
      role: 'Officer',
      status: 'active',
      lastLogin: '30 minutes ago',
      phone: '+1 (555) 678-9012',
      incidents: 78,
    },
    {
      id: 7,
      name: 'Linda Garcia',
      email: 'linda.garcia@vigil.com',
      role: 'Security',
      status: 'active',
      lastLogin: '15 minutes ago',
      phone: '+1 (555) 789-0123',
      incidents: 34,
    },
    {
      id: 8,
      name: 'Robert Brown',
      email: 'robert.brown@vigil.com',
      role: 'Officer',
      status: 'inactive',
      lastLogin: '2 days ago',
      phone: '+1 (555) 890-1234',
      incidents: 52,
    },
  ];

  const filteredUsers = users.filter((user) => {
    if (roleFilter !== 'all' && user.role !== roleFilter) return false;
    if (
      searchQuery &&
      !user.name.toLowerCase().includes(searchQuery.toLowerCase()) &&
      !user.email.toLowerCase().includes(searchQuery.toLowerCase())
    ) {
      return false;
    }
    return true;
  });

  const getRoleIcon = (role: string) => {
    switch (role) {
      case 'Admin':
        return <Shield className="w-4 h-4" />;
      case 'Officer':
        return <Eye className="w-4 h-4" />;
      case 'Security':
        return <Lock className="w-4 h-4" />;
      default:
        return <Users className="w-4 h-4" />;
    }
  };

  const getRoleBadgeStyle = (role: string) => {
    switch (role) {
      case 'Admin':
        return 'bg-blue-100 text-blue-800 border-blue-200 dark:bg-blue-500/20 dark:text-blue-400 dark:border-blue-500/30';
      case 'Officer':
        return 'bg-amber-100 text-amber-800 border-amber-200 dark:bg-amber-500/20 dark:text-amber-400 dark:border-amber-500/30';
      case 'Security':
        return 'bg-purple-100 text-purple-800 border-purple-200 dark:bg-purple-500/20 dark:text-purple-400 dark:border-purple-500/30';
      default:
        return 'bg-gray-100 text-gray-800 border-gray-200 dark:bg-gray-500/20 dark:text-gray-400 dark:border-gray-500/30';
    }
  };

  const getInitials = (name: string) => {
    return name
      .split(' ')
      .map((n) => n[0])
      .join('')
      .toUpperCase();
  };

  const handleEditUser = (userId: number) => {
    toast.info(`Editing user ${userId}`);
  };

  const handleDeleteUser = (userId: number) => {
    toast.success(`User ${userId} deleted`);
  };

  // Mobile-optimized user card component
  const MobileUserCard = ({ user }: { user: typeof users[0] }) => (
    <Card className="mb-3 hover:bg-muted/50 transition-colors">
      <CardContent className="p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3 flex-1 min-w-0">
            <Avatar className="w-12 h-12 bg-gradient-to-br from-blue-500 to-blue-600 flex-shrink-0">
              <AvatarFallback className="bg-transparent text-white text-sm">
                {getInitials(user.name)}
              </AvatarFallback>
            </Avatar>
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 mb-1">
                <p className="font-semibold text-sm truncate">{user.name}</p>
                <Badge 
                  className={`${getRoleBadgeStyle(user.role)} text-xs px-2 py-0.5`}
                >
                  {getRoleIcon(user.role)}
                  <span className="ml-1 hidden sm:inline">{user.role}</span>
                  <span className="ml-1 sm:hidden">{user.role.charAt(0)}</span>
                </Badge>
              </div>
              <p className="text-xs text-muted-foreground truncate">{user.email}</p>
              <div className="flex items-center gap-3 mt-2 text-xs">
                <span className="text-muted-foreground truncate">
                  📞 {user.phone.replace(/\D/g, '').slice(-10)}
                </span>
                <Badge
                  className={
                    user.status === 'active'
                      ? 'bg-green-100 text-green-800 border-green-200 dark:bg-green-500/20 dark:text-green-400 dark:border-green-500/30 text-xs px-2 py-0'
                      : 'bg-gray-100 text-gray-800 border-gray-200 dark:bg-gray-500/20 dark:text-gray-400 dark:border-gray-500/30 text-xs px-2 py-0'
                  }
                >
                  {user.status}
                </Badge>
              </div>
            </div>
          </div>
          <Sheet>
            <SheetTrigger asChild>
              <Button variant="ghost" size="icon" className="ml-2">
                <ChevronRight className="h-4 w-4" />
              </Button>
            </SheetTrigger>
            <SheetContent side="bottom" className="h-[80vh]">
              <SheetHeader>
                <SheetTitle>User Details</SheetTitle>
              </SheetHeader>
              <div className="mt-6 space-y-4">
                <div className="flex items-center gap-4">
                  <Avatar className="w-16 h-16 bg-gradient-to-br from-blue-500 to-blue-600">
                    <AvatarFallback className="bg-transparent text-white text-lg">
                      {getInitials(user.name)}
                    </AvatarFallback>
                  </Avatar>
                  <div>
                    <h3 className="text-lg font-semibold">{user.name}</h3>
                    <p className="text-sm text-muted-foreground">{user.email}</p>
                  </div>
                </div>
                
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <Label className="text-xs text-muted-foreground">Role</Label>
                    <Badge className={`${getRoleBadgeStyle(user.role)} mt-1`}>
                      {getRoleIcon(user.role)}
                      <span className="ml-1">{user.role}</span>
                    </Badge>
                  </div>
                  <div>
                    <Label className="text-xs text-muted-foreground">Status</Label>
                    <Badge className={`${
                      user.status === 'active'
                        ? 'bg-green-100 text-green-800 border-green-200 dark:bg-green-500/20 dark:text-green-400 dark:border-green-500/30 mt-1'
                        : 'bg-gray-100 text-gray-800 border-gray-200 dark:bg-gray-500/20 dark:text-gray-400 dark:border-gray-500/30 mt-1'
                    }`}>
                      {user.status}
                    </Badge>
                  </div>
                </div>
                
                <div className="space-y-2">
                  <Label className="text-xs text-muted-foreground">Phone</Label>
                  <p className="text-sm">{user.phone}</p>
                </div>
                
                <div className="space-y-2">
                  <Label className="text-xs text-muted-foreground">Last Login</Label>
                  <p className="text-sm">{user.lastLogin}</p>
                </div>
                
                <div className="space-y-2">
                  <Label className="text-xs text-muted-foreground">Incidents Handled</Label>
                  <p className="text-sm font-semibold">{user.incidents}</p>
                </div>
                
                <div className="flex gap-2 pt-4">
                  <Button 
                    size="sm" 
                    variant="outline" 
                    className="flex-1"
                    onClick={() => handleEditUser(user.id)}
                  >
                    <Edit className="h-4 w-4 mr-2" />
                    Edit
                  </Button>
                  <Button 
                    size="sm" 
                    variant="destructive" 
                    className="flex-1"
                    onClick={() => handleDeleteUser(user.id)}
                  >
                    <Trash2 className="h-4 w-4 mr-2" />
                    Delete
                  </Button>
                </div>
              </div>
            </SheetContent>
          </Sheet>
        </div>
        
        <div className="flex items-center justify-between mt-3 pt-3 border-t">
          <div className="text-xs text-muted-foreground">
            Last login: {user.lastLogin}
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs text-muted-foreground">
              Incidents: <span className="font-semibold">{user.incidents}</span>
            </span>
            <div className="flex gap-1">
              <Button
                size="icon"
                variant="ghost"
                className="w-6 h-6 text-muted-foreground hover:text-foreground"
                onClick={() => handleEditUser(user.id)}
              >
                <Edit className="w-3 h-3" />
              </Button>
              <Button
                size="icon"
                variant="ghost"
                className="w-6 h-6 text-muted-foreground hover:text-red-600 dark:hover:text-red-400"
                onClick={() => handleDeleteUser(user.id)}
              >
                <Trash2 className="w-3 h-3" />
              </Button>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );

  return (
    <div className="space-y-4 md:space-y-6">
      {/* Header - Optimized for mobile */}
      <div className="flex flex-col gap-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-xl md:text-2xl font-bold text-foreground">User Management</h2>
            <p className="text-xs md:text-sm text-muted-foreground">
              {filteredUsers.length} users • Manage system access
            </p>
          </div>
          <Dialog>
            <DialogTrigger asChild>
              <Button size="sm" className="bg-blue-600 hover:bg-blue-700 text-white">
                <Plus className="w-4 h-4" />
                <span className="hidden sm:inline ml-2">Add User</span>
              </Button>
            </DialogTrigger>
            <DialogContent className="bg-card border-border text-foreground max-w-[95vw] rounded-lg">
              <DialogHeader>
                <DialogTitle className="text-lg">Add New User</DialogTitle>
                <DialogDescription className="text-muted-foreground text-sm">
                  Create a new user account
                </DialogDescription>
              </DialogHeader>
              <div className="space-y-3 mt-4">
                <div className="space-y-1.5">
                  <Label className="text-foreground text-sm">Full Name</Label>
                  <Input
                    placeholder="Full name"
                    className="bg-background border-border text-foreground text-sm h-9"
                  />
                </div>
                <div className="space-y-1.5">
                  <Label className="text-foreground text-sm">Email</Label>
                  <Input
                    type="email"
                    placeholder="user@vigil.com"
                    className="bg-background border-border text-foreground text-sm h-9"
                  />
                </div>
                <div className="space-y-1.5">
                  <Label className="text-foreground text-sm">Phone</Label>
                  <Input
                    type="tel"
                    placeholder="Phone number"
                    className="bg-background border-border text-foreground text-sm h-9"
                  />
                </div>
                <div className="space-y-1.5">
                  <Label className="text-foreground text-sm">Role</Label>
                  <Select>
                    <SelectTrigger className="bg-background border-border text-foreground text-sm h-9">
                      <SelectValue placeholder="Select role" />
                    </SelectTrigger>
                    <SelectContent className="bg-card border-border text-foreground">
                      <SelectItem value="admin">Admin</SelectItem>
                      <SelectItem value="officer">Officer</SelectItem>
                      <SelectItem value="security">Security</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <Button className="w-full bg-blue-600 hover:bg-blue-700 text-white h-9">
                  Create User
                </Button>
              </div>
            </DialogContent>
          </Dialog>
        </div>
      </div>

      {/* Stats Cards - Horizontal scroll on mobile */}
      <div className="overflow-x-auto pb-2 -mx-4 px-4 md:mx-0 md:px-0">
        <div className="flex md:grid md:grid-cols-4 gap-3 min-w-max md:min-w-0">
          <Card className="min-w-[150px] md:min-w-0">
            <CardContent className="p-3">
              <div className="flex items-center gap-2">
                <div className="w-8 h-8 rounded-lg bg-blue-500/20 flex items-center justify-center flex-shrink-0">
                  <Users className="w-4 h-4 text-blue-600 dark:text-blue-400" />
                </div>
                <div>
                  <div className="text-lg font-semibold text-foreground">8</div>
                  <div className="text-xs text-muted-foreground">Total Users</div>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="min-w-[150px] md:min-w-0">
            <CardContent className="p-3">
              <div className="flex items-center gap-2">
                <div className="w-8 h-8 rounded-lg bg-green-500/20 flex items-center justify-center flex-shrink-0">
                  <Shield className="w-4 h-4 text-green-600 dark:text-green-400" />
                </div>
                <div>
                  <div className="text-lg font-semibold text-foreground">7</div>
                  <div className="text-xs text-muted-foreground">Active</div>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="min-w-[150px] md:min-w-0">
            <CardContent className="p-3">
              <div className="flex items-center gap-2">
                <div className="w-8 h-8 rounded-lg bg-amber-500/20 flex items-center justify-center flex-shrink-0">
                  <Eye className="w-4 h-4 text-amber-600 dark:text-amber-400" />
                </div>
                <div>
                  <div className="text-lg font-semibold text-foreground">4</div>
                  <div className="text-xs text-muted-foreground">Officers</div>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="min-w-[150px] md:min-w-0">
            <CardContent className="p-3">
              <div className="flex items-center gap-2">
                <div className="w-8 h-8 rounded-lg bg-purple-500/20 flex items-center justify-center flex-shrink-0">
                  <Lock className="w-4 h-4 text-purple-600 dark:text-purple-400" />
                </div>
                <div>
                  <div className="text-lg font-semibold text-foreground">2</div>
                  <div className="text-xs text-muted-foreground">Security</div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Filters - Stacked on mobile */}
      <Card className="bg-card/80 backdrop-blur border-border">
        <CardContent className="p-3">
          <div className="flex flex-col gap-3">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
              <Input
                placeholder="Search users..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10 bg-background border-border text-foreground placeholder:text-muted-foreground h-9 text-sm"
              />
            </div>
            <Select value={roleFilter} onValueChange={setRoleFilter}>
              <SelectTrigger className="bg-background border-border text-foreground h-9 text-sm">
                <SelectValue placeholder="All Roles" />
              </SelectTrigger>
              <SelectContent className="bg-card border-border text-foreground">
                <SelectItem value="all">All Roles</SelectItem>
                <SelectItem value="Admin">Admin</SelectItem>
                <SelectItem value="Officer">Officer</SelectItem>
                <SelectItem value="Security">Security</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      {/* User List/Table - Mobile cards, Desktop table */}
      <Card className="bg-card/80 backdrop-blur border-border">
        <CardHeader className="p-4 pb-2">
          <CardTitle className="flex items-center gap-2 text-card-foreground text-lg">
            <Users className="w-5 h-5 text-blue-600 dark:text-blue-400" />
            User List
            <Badge variant="outline" className="ml-2">
              {filteredUsers.length}
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          {isMobileView ? (
            // Mobile Cards View
            <ScrollArea className="h-[calc(100vh-350px)] md:h-[500px] p-4">
              <div className="space-y-2">
                {filteredUsers.map((user) => (
                  <MobileUserCard key={user.id} user={user} />
                ))}
              </div>
            </ScrollArea>
          ) : (
            // Desktop Table View
            <ScrollArea className="h-[500px]">
              <Table className="border-collapse">
                <TableHeader className="sticky top-0 bg-card z-10">
                  <TableRow className="border-border hover:bg-transparent">
                    <TableHead className="text-muted-foreground text-xs py-3">User</TableHead>
                    <TableHead className="text-muted-foreground text-xs py-3">Role</TableHead>
                    <TableHead className="text-muted-foreground text-xs py-3">Contact</TableHead>
                    <TableHead className="text-muted-foreground text-xs py-3">Status</TableHead>
                    <TableHead className="text-muted-foreground text-xs py-3">Last Login</TableHead>
                    <TableHead className="text-muted-foreground text-xs py-3">Incidents</TableHead>
                    <TableHead className="text-muted-foreground text-xs py-3">Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {filteredUsers.map((user) => (
                    <TableRow key={user.id} className="border-border hover:bg-muted/50">
                      <TableCell className="py-3">
                        <div className="flex items-center gap-3">
                          <Avatar className="w-9 h-9 bg-gradient-to-br from-blue-500 to-blue-600">
                            <AvatarFallback className="bg-transparent text-white text-sm">
                              {getInitials(user.name)}
                            </AvatarFallback>
                          </Avatar>
                          <div>
                            <p className="text-foreground text-sm font-medium">{user.name}</p>
                            <p className="text-xs text-muted-foreground truncate max-w-[180px]">{user.email}</p>
                          </div>
                        </div>
                      </TableCell>
                      <TableCell className="py-3">
                        <Badge className={`${getRoleBadgeStyle(user.role)} text-xs`}>
                          {getRoleIcon(user.role)}
                          <span className="ml-1">{user.role}</span>
                        </Badge>
                      </TableCell>
                      <TableCell className="py-3 text-muted-foreground text-xs">
                        <div className="flex items-center gap-2">
                          <Phone className="w-3 h-3" />
                          <span className="truncate max-w-[120px]">{user.phone}</span>
                        </div>
                      </TableCell>
                      <TableCell className="py-3">
                        <Badge
                          className={
                            user.status === 'active'
                              ? 'bg-green-100 text-green-800 border-green-200 dark:bg-green-500/20 dark:text-green-400 dark:border-green-500/30 text-xs'
                              : 'bg-gray-100 text-gray-800 border-gray-200 dark:bg-gray-500/20 dark:text-gray-400 dark:border-gray-500/30 text-xs'
                          }
                        >
                          {user.status}
                        </Badge>
                      </TableCell>
                      <TableCell className="py-3 text-muted-foreground text-xs">
                        {user.lastLogin}
                      </TableCell>
                      <TableCell className="py-3 text-foreground text-sm font-medium">
                        {user.incidents}
                      </TableCell>
                      <TableCell className="py-3">
                        <div className="flex items-center gap-1">
                          <Button
                            size="icon"
                            variant="ghost"
                            className="w-7 h-7 text-muted-foreground hover:text-foreground"
                            onClick={() => handleEditUser(user.id)}
                          >
                            <Edit className="w-3.5 h-3.5" />
                          </Button>
                          <Button
                            size="icon"
                            variant="ghost"
                            className="w-7 h-7 text-muted-foreground hover:text-red-600 dark:hover:text-red-400"
                            onClick={() => handleDeleteUser(user.id)}
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                          </Button>
                        </div>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </ScrollArea>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
