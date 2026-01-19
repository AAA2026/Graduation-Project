import { ChevronRight, Home } from 'lucide-react';

export interface BreadcrumbItem {
  label: string;
  onClick?: () => void;
  icon?: React.ReactNode;
}

interface BreadcrumbProps {
  items: BreadcrumbItem[];
}

export default function Breadcrumb({ items }: BreadcrumbProps) {
  return (
    <nav className="flex items-center space-x-2 text-sm mb-6">
      {/* Home Icon */}
      <button
        onClick={items[0]?.onClick}
        className="flex items-center text-[var(--vigil-text-secondary)] hover:text-[var(--vigil-primary)] transition-colors"
        aria-label="Home"
      >
        <Home className="w-4 h-4" />
      </button>

      {/* Breadcrumb Items */}
      {items.map((item, index) => (
        <div key={index} className="flex items-center space-x-2">
          <ChevronRight className="w-4 h-4 text-[var(--vigil-text-muted)]" />
          
          {item.onClick ? (
            <button
              onClick={item.onClick}
              className="flex items-center space-x-1 text-[var(--vigil-text-secondary)] hover:text-[var(--vigil-primary)] transition-colors"
            >
              {item.icon && <span>{item.icon}</span>}
              <span>{item.label}</span>
            </button>
          ) : (
            <span className="flex items-center space-x-1 text-gray-900 dark:text-white font-medium">
              {item.icon && <span>{item.icon}</span>}
              <span>{item.label}</span>
            </span>
          )}
        </div>
      ))}
    </nav>
  );
}
