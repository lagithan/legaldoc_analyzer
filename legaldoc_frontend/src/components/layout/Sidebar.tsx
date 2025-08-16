import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  Home,
  FileText,
  BarChart3,
  Upload,
  HelpCircle,
  Settings
} from 'lucide-react';
import { cn } from '../../utils/cn';
import { ROUTES } from '../../utils/constants';

export interface SidebarProps {
  className?: string;
  isCollapsed?: boolean;
}

const Sidebar: React.FC<SidebarProps> = ({
  className,
  isCollapsed = false
}) => {
  const navigationItems = [
    {
      label: 'Dashboard',
      icon: Home,
      href: ROUTES.DASHBOARD,
      description: 'Overview and recent activity'
    },
    {
      label: 'Documents',
      icon: FileText,
      href: ROUTES.DOCUMENTS,
      description: 'Manage your legal documents'
    },
    {
      label: 'Analytics',
      icon: BarChart3,
      href: ROUTES.ANALYTICS,
      description: 'Risk analysis and insights'
    }
  ];

  return (
    <aside className={cn(
      'bg-white border-r border-pearl-gray transition-all duration-300',
      isCollapsed ? 'w-16' : 'w-64',
      className
    )}>
      <div className="h-full flex flex-col">
        {/* Navigation */}
        <nav className="flex-1 p-4">
          <div className="space-y-2">
            {/* Main Navigation */}
            <div className="space-y-1">
              {navigationItems.map((item) => {
                const Icon = item.icon;
                return (
                  <NavLink
                    key={item.href}
                    to={item.href}
                    className={({ isActive }) =>
                      cn(
                        'sidebar-item group',
                        isActive && 'active'
                      )
                    }
                    title={isCollapsed ? item.label : undefined}
                  >
                    <Icon className="w-5 h-5 flex-shrink-0" />
                    {!isCollapsed && (
                      <div className="ml-3 min-w-0 flex-1">
                        <div className="text-sm font-medium">
                          {item.label}
                        </div>
                        <div className="text-xs text-gray-500 group-hover:text-gray-600 transition-colors">
                          {item.description}
                        </div>
                      </div>
                    )}
                  </NavLink>
                );
              })}
            </div>

            {/* Divider */}
            {!isCollapsed && (
              <div className="py-4">
                <div className="border-t border-pearl-gray" />
              </div>
            )}
          </div>
        </nav>

        {/* Footer */}
        {!isCollapsed && (
          <div className="p-4 border-t border-pearl-gray">
            <div className="text-xs text-gray-500 space-y-1">
              <div className="font-medium">AI Legal Assistant</div>
              <div>Version 8.0.0</div>
              <div>Powered by Legal-BERT</div>
            </div>
          </div>
        )}
      </div>
    </aside>
  );
};

export default Sidebar;