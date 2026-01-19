import { useState } from 'react';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Shield, Eye, Lock } from 'lucide-react';
import { toast } from 'sonner';
import { VigilLogo } from './VigilLogo';
import { ThemeToggle } from './ThemeToggle';
import { useTheme } from './ThemeProvider';
import { AnimatedBackgroundBeams } from './AnimatedBackgroundBeams';

type UserRole = 'admin' | 'officer' | 'security';

interface LoginScreenProps {
  onLogin: (role: UserRole) => void;
}

export function LoginScreen({ onLogin }: LoginScreenProps) {
  const defaultEmails: Record<UserRole, string> = {
    admin: 'admin@vigil.com',
    officer: 'officer@vigil.com',
    security: 'security@vigil.com',
  };
  const [roleTab, setRoleTab] = useState<UserRole>('admin');
  const [email, setEmail] = useState(defaultEmails['admin']);
  const [password, setPassword] = useState('');
  const { theme } = useTheme();
  const [modal, setModal] = useState<null | 'about' | 'help' | 'more'>(null);

  // When roleTab changes, auto-fill email

  const handleTabChange = (role: UserRole) => {
    setRoleTab(role);
    setEmail(defaultEmails[role]);
    setPassword('');
  };

  const handleLogin = async (role: UserRole) => {
    if (!email || !password) {
      toast.error('Please enter email and password');
      return;
    }
    // Fix ImportMeta typing for Vite
    const apiUrl = (import.meta as ImportMeta & { env: { VITE_API_URL?: string } }).env.VITE_API_URL || 'http://localhost:5000';
    try {
      const response = await fetch(`${apiUrl}/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password, role })
      });
      const data = await response.json();
      if (response.ok && data.token) {
        localStorage.setItem('authToken', data.token);
        localStorage.setItem('userRole', data.role || role);
        toast.success(`Logged in as ${data.role || role}`);
        onLogin(data.role || role);
        // loggedIn variable is not defined or used, remove this line
      } else {
        toast.error(data.message || 'Login failed');
      }
    } catch (error) {
      // Backend unreachable: allow local login in explicit OFFLINE mode
      console.error("Login error:", error);
      localStorage.setItem('authToken', 'OFFLINE');
      localStorage.setItem('userRole', role);
      localStorage.setItem('offlineMode', 'true');
      toast.warning(`Backend unreachable: ${(error as Error).message}. Logged in as OFFLINE mode.`);
      onLogin(role);
    }
  };

  return (
    <AnimatedBackgroundBeams mode={theme === 'dark' ? 'dark' : 'light'}>
      <div className="min-h-screen flex items-center justify-center p-4 relative">
        {/* Content Layer */}
        <div className="relative z-10 w-full max-w-md">
          {/* Theme Toggle - Top Right */}
          <div className="fixed top-6 right-6 z-20">
            <ThemeToggle />
            </div>

          {/* Logo and Header */}
          <div className="text-center mb-8">
            <div className="inline-flex items-center justify-center mb-4">
              <VigilLogo width={270} height={90} theme={theme} />
            </div>
            <p className="text-gray-600 dark:text-gray-400">AI-Powered Surveillance System</p>
          </div>

          <Card className="border-gray-200 dark:border-gray-800 bg-white/80 dark:bg-[#13182b]/80 backdrop-blur-xl shadow-2xl">
            <CardHeader>
              <CardTitle className="text-gray-900 dark:text-white text-2xl md:text-3xl font-bold text-center w-full">Sign In</CardTitle>
              <CardDescription className="text-center w-full">Choose your role and sign in to continue</CardDescription>
            </CardHeader>
            <CardContent>
              <Tabs defaultValue="admin" value={roleTab} onValueChange={handleTabChange} className="w-full">
                <TabsList className="grid w-full grid-cols-3 bg-gray-100 dark:bg-[#0a0e1a]">
                  <TabsTrigger value="admin" className="data-[state=active]:bg-blue-500 dark:data-[state=active]:bg-cyan-500 data-[state=active]:text-white">
                    Admin
                  </TabsTrigger>
                  <TabsTrigger value="officer" className="data-[state=active]:bg-blue-500 dark:data-[state=active]:bg-cyan-500 data-[state=active]:text-white">
                    Officer
                  </TabsTrigger>
                  <TabsTrigger value="security" className="data-[state=active]:bg-blue-500 dark:data-[state=active]:bg-cyan-500 data-[state=active]:text-white">
                    Security
                  </TabsTrigger>
                </TabsList>

                <TabsContent value="admin" className="space-y-4">
                  <LoginForm
                    role="admin"
                    icon={<Shield className="w-5 h-5" />}
                    email={email}
                    password={password}
                    onEmailChange={setEmail}
                    onPasswordChange={setPassword}
                    onSubmit={() => handleLogin('admin')}
                  />
                </TabsContent>
                <TabsContent value="officer" className="space-y-4">
                  <LoginForm
                    role="officer"
                    icon={<Eye className="w-5 h-5" />}
                    email={email}
                    password={password}
                    onEmailChange={setEmail}
                    onPasswordChange={setPassword}
                    onSubmit={() => handleLogin('officer')}
                  />
                </TabsContent>
                <TabsContent value="security" className="space-y-4">
                  <LoginForm
                    role="security"
                    icon={<Lock className="w-5 h-5" />}
                    email={email}
                    password={password}
                    onEmailChange={setEmail}
                    onPasswordChange={setPassword}
                    onSubmit={() => handleLogin('security')}
                  />
                </TabsContent>
              </Tabs>
            </CardContent>
          </Card>

        {/* About, Help, More Section (Facebook-style small links) */}
        <div className="mt-8 w-full max-w-md mx-auto text-center z-10">
            <div className="flex flex-row justify-center gap-4 text-sm text-gray-500 dark:text-gray-400">
              <button
                className="bg-transparent border-none p-0 m-0 underline hover:text-blue-600 dark:hover:text-cyan-400 cursor-pointer focus:outline-none"
                style={{ fontSize: '15px', fontWeight: 400, background: 'none' }}
                onClick={() => setModal('about')}
                type="button"
              >
                About Vigil
              </button>
              <span className="mx-1">·</span>
              <button
                className="bg-transparent border-none p-0 m-0 underline hover:text-blue-600 dark:hover:text-cyan-400 cursor-pointer focus:outline-none"
                style={{ fontSize: '15px', fontWeight: 400, background: 'none' }}
                onClick={() => setModal('help')}
                type="button"
              >
                Help
              </button>
              <span className="mx-1">·</span>
              <button
                className="bg-transparent border-none p-0 m-0 underline hover:text-blue-600 dark:hover:text-cyan-400 cursor-pointer focus:outline-none"
                style={{ fontSize: '15px', fontWeight: 400, background: 'none' }}
                onClick={() => setModal('more')}
                type="button"
              >
                More
              </button>
            </div>
            {modal && (
              <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40">
                <div className="bg-white dark:bg-[#18181b] rounded-xl shadow-2xl p-8 max-w-md w-full text-center border border-gray-200 dark:border-gray-800">
                  <h2 className="text-xl font-bold mb-4 text-blue-600 dark:text-cyan-400">
                    {modal === 'about' && 'About Vigil'}
                    {modal === 'help' && 'Help'}
                    {modal === 'more' && 'More'}
                  </h2>
                  <div className="text-gray-700 dark:text-gray-300 mb-6">
                    {modal === 'about' && (
                      <>
                        Vigil is an AI-powered surveillance platform for real-time incident detection and security analytics.<br />
                        Built for modern security teams, Vigil leverages advanced models to detect violence, crashes, and people in live video feeds.
                      </>
                    )}
                    {modal === 'help' && (
                      <>
                        Need assistance? Contact our support team at <a href="mailto:support@vigil.com" className="underline text-blue-500 dark:text-cyan-400">support@vigil.com</a>.<br />
                        For documentation and FAQs, visit the Help Center from the main dashboard after login.
                      </>
                    )}
                    {modal === 'more' && (
                      <>
                        Read our <a href="#" className="underline text-blue-500 dark:text-cyan-400">Privacy Policy</a> and <a href="#" className="underline text-blue-500 dark:text-cyan-400">Terms of Service</a>.<br />
                        Vigil is committed to protecting your data and privacy.
                      </>
                    )}
                  </div>
                  <button
                    className="px-6 py-2 rounded bg-blue-500 hover:bg-blue-600 dark:bg-cyan-500 dark:hover:bg-cyan-600 text-white font-semibold"
                    onClick={() => setModal(null)}
                    type="button"
                  >
                    Close
                  </button>
                </div>
              </div>
            )}
        </div>
      </div>
    </div>
  </AnimatedBackgroundBeams>
  );
}

interface LoginFormProps {
  role: string;
  icon: React.ReactNode;
  email: string;
  password: string;
  onEmailChange: (value: string) => void;
  onPasswordChange: (value: string) => void;
  onSubmit: () => void;
}

function LoginForm({ role, icon, email, password, onEmailChange, onPasswordChange, onSubmit }: LoginFormProps) {
  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        onSubmit();
      }}
      className="space-y-4"
    >
      <div className="flex items-center gap-2 text-blue-500 dark:text-cyan-400 mb-4">
        {icon}
        <span className="capitalize">Sign in as {role}</span>
      </div>

      <div className="space-y-2">
        <Label htmlFor={`email-${role}`} className="text-gray-700 dark:text-gray-300">
          Email
        </Label>
        <Input
          id={`email-${role}`}
          type="email"
          placeholder={`${role}@vigil.com`}
          value={email}
          onChange={(e) => onEmailChange(e.target.value)}
          className="bg-gray-50 dark:bg-[#0a0e1a] border-gray-300 dark:border-gray-700 text-gray-900 dark:text-white placeholder:text-gray-400 dark:placeholder:text-gray-500"
        />
      </div>

      <div className="space-y-2">
        <Label htmlFor={`password-${role}`} className="text-gray-700 dark:text-gray-300">
          Password
        </Label>
        <Input
          id={`password-${role}`}
          type="password"
          placeholder="Enter your password"
          value={password}
          onChange={(e) => onPasswordChange(e.target.value)}
          className="bg-gray-50 dark:bg-[#0a0e1a] border-gray-300 dark:border-gray-700 text-gray-900 dark:text-white placeholder:text-gray-400 dark:placeholder:text-gray-500"
        />
      </div>

      <Button type="submit" className="w-full bg-blue-500 hover:bg-blue-600 dark:bg-cyan-500 dark:hover:bg-cyan-600 text-white">
        Sign In
      </Button>
    </form>
  );
}