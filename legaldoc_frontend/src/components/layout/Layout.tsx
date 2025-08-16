import React, { useState } from 'react';
import { Outlet } from 'react-router-dom';
import Header from './Header';
import Sidebar from './Sidebar';
import { cn } from '../../utils/cn';

export interface LayoutProps {
  className?: string;
}

const Layout: React.FC<LayoutProps> = ({ className }) => {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  return (
    <div className={cn('min-h-screen bg-pearl-white', className)}>
      {/* Header */}
      <Header />

      <div className="flex h-[calc(100vh-73px)]">
        {/* Sidebar */}
        <Sidebar
          isCollapsed={sidebarCollapsed}
        />

        {/* Main Content */}
        <main className="flex-1 overflow-auto">
          <div className="p-6">
            <Outlet />
          </div>
        </main>
      </div>

      {/* Chat Assistant - positioned absolutely */}
     
    </div>
  );
};

export default Layout;