import { SidebarProvider, SidebarTrigger } from "@/components/ui/sidebar";
import { ReactNode, useEffect } from "react";
import { AppSidebar } from "../AppSidebar";

export function AppLayout({ children }: { children: ReactNode }) {
  useEffect(() => {
    document.title = "AI Legal Document Analyzer";
  }, []);

  return (
    <SidebarProvider>
      <AppSidebar />
      <main className="flex-1 flex flex-col min-h-screen w-full">
        <header className="h-14 border-b flex items-center px-3 gap-3 bg-background/80 backdrop-blur supports-[backdrop-filter]:bg-background/60">
          <SidebarTrigger />
          <h1 className="text-lg font-semibold tracking-tight">AI Legal Document Analyzer</h1>
        </header>
        <div className="flex-1 p-2">
          {children}
        </div>
      </main>
    </SidebarProvider>
  );
}