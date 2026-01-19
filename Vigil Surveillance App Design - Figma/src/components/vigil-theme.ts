/**
 * Vigil Theme System
 * 
 * Centralized color utilities for consistent theming across the application.
 * Use these helper functions to ensure proper light/dark mode color application.
 */

export const vigilColors = {
  light: {
    // Primary colors
    primary: '#3b82f6',        // Light blue
    primaryHover: '#2563eb',   // Darker blue
    secondary: '#06b6d4',      // Cyan accent
    
    // Backgrounds
    bg: '#f8fafc',             // Off-white
    bgCard: '#ffffff',         // White cards
    bgCardHover: '#f1f5f9',    // Slightly darker on hover
    
    // Text colors
    textPrimary: '#0f172a',    // Dark slate
    textSecondary: '#475569',  // Gray
    textMuted: '#64748b',      // Muted gray
    
    // Borders
    border: '#e2e8f0',         // Light border
    borderHover: '#cbd5e1',    // Darker on hover
    
    // Status colors
    success: '#10b981',        // Green
    warning: '#f59e0b',        // Amber
    danger: '#ef4444',         // Red
    info: '#3b82f6',           // Blue
  },
  dark: {
    // Primary colors
    primary: '#06b6d4',        // Electric cyan
    primaryHover: '#0891b2',   // Darker cyan
    secondary: '#f59e0b',      // Amber accent
    
    // Backgrounds
    bg: '#0a0e1a',             // Dark charcoal
    bgCard: '#13182b',         // Slightly lighter card
    bgCardHover: '#1e293b',    // Lighter on hover
    
    // Text colors
    textPrimary: '#f8fafc',    // Off-white
    textSecondary: '#cbd5e1',  // Light gray
    textMuted: '#94a3b8',      // Muted gray
    
    // Borders
    border: '#1e293b',         // Dark border
    borderHover: '#334155',    // Lighter on hover
    
    // Status colors
    success: '#10b981',        // Green
    warning: '#f59e0b',        // Amber
    danger: '#ef4444',         // Red
    info: '#06b6d4',           // Cyan
  },
};

/**
 * Tailwind class utilities for common theme patterns
 */
export const vigilClasses = {
  // Backgrounds
  bg: 'bg-gray-50 dark:bg-[#0a0e1a]',
  bgCard: 'bg-white dark:bg-[#13182b]',
  bgCardHover: 'hover:bg-gray-50 dark:hover:bg-[#1e293b]',
  
  // Text
  textPrimary: 'text-gray-900 dark:text-white',
  textSecondary: 'text-gray-600 dark:text-gray-300',
  textMuted: 'text-gray-500 dark:text-gray-400',
  
  // Borders
  border: 'border-gray-200 dark:border-gray-800',
  borderHover: 'hover:border-gray-300 dark:hover:border-gray-700',
  
  // Buttons - Primary
  btnPrimary: 'bg-blue-500 hover:bg-blue-600 dark:bg-cyan-500 dark:hover:bg-cyan-600 text-white',
  btnPrimaryOutline: 'border-blue-500 dark:border-cyan-500 text-blue-500 dark:text-cyan-500 hover:bg-blue-50 dark:hover:bg-cyan-950',
  
  // Buttons - Secondary
  btnSecondary: 'bg-gray-200 hover:bg-gray-300 dark:bg-gray-700 dark:hover:bg-gray-600 text-gray-900 dark:text-white',
  
  // Buttons - Ghost
  btnGhost: 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white hover:bg-gray-100 dark:hover:bg-gray-800',
  
  // Status badges
  badgeSuccess: 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400 border-green-200 dark:border-green-800',
  badgeWarning: 'bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-400 border-amber-200 dark:border-amber-800',
  badgeDanger: 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400 border-red-200 dark:border-red-800',
  badgeInfo: 'bg-blue-100 dark:bg-cyan-900/30 text-blue-700 dark:text-cyan-400 border-blue-200 dark:border-cyan-800',
  
  // Cards
  card: 'bg-white dark:bg-[#13182b] border-gray-200 dark:border-gray-800 rounded-lg shadow-sm',
  cardHover: 'bg-white dark:bg-[#13182b] border-gray-200 dark:border-gray-800 rounded-lg shadow-sm hover:shadow-md transition-shadow',
  
  // Inputs
  input: 'bg-gray-50 dark:bg-[#0a0e1a] border-gray-300 dark:border-gray-700 text-gray-900 dark:text-white placeholder:text-gray-400 dark:placeholder:text-gray-500',
  
  // Charts (for recharts)
  chartGrid: 'stroke-gray-200 dark:stroke-gray-800',
  chartText: 'fill-gray-600 dark:fill-gray-400',
};

/**
 * Get color for current theme
 */
export function getVigilColor(
  colorKey: keyof typeof vigilColors.light,
  theme: 'light' | 'dark'
): string {
  return vigilColors[theme][colorKey];
}

/**
 * Chart color palette for recharts
 */
export const chartColors = {
  light: {
    primary: '#3b82f6',
    secondary: '#06b6d4',
    tertiary: '#8b5cf6',
    quaternary: '#10b981',
    quinary: '#f59e0b',
  },
  dark: {
    primary: '#06b6d4',
    secondary: '#8b5cf6',
    tertiary: '#10b981',
    quaternary: '#f59e0b',
    quinary: '#ef4444',
  },
};

/**
 * Get chart colors for current theme
 */
export function getChartColors(theme: 'light' | 'dark'): string[] {
  return Object.values(chartColors[theme]);
}
