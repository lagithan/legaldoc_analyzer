import { useEffect, useMemo, useRef, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { useMutation, useQuery } from "@tanstack/react-query";
import { askQuestion, getDocument, suggestQuestions } from "@/lib/api";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { Progress } from "@/components/ui/progress";
import { motion, AnimatePresence } from "framer-motion";
import { useToast } from "@/hooks/use-toast";
import { Loader2, FileText, AlertTriangle, MessageSquare, Bot, Shield, TrendingUp, Download, ArrowLeft, Send, Sparkles } from 'lucide-react';
import jsPDF from 'jspdf';

export default function DocumentDetail() {
  const { id = "" } = useParams();
  const navigate = useNavigate();
  const { toast } = useToast();
  const [question, setQuestion] = useState("");
  const [chat, setChat] = useState<
    Array<{ role: "user" | "assistant"; content: string; sections?: any[]; timestamp?: string }>
  >([]);
  const bottomRef = useRef<HTMLDivElement>(null);

  const handleAsk = (q: string) => {
    setQuestion(q);
    if (!qMutation.isPending) {
      qMutation.mutate();
    }
  };

  const { data, isLoading, error } = useQuery({
    queryKey: ["document", id],
    queryFn: () => getDocument(id),
  });

  const { data: sugg } = useQuery({
    queryKey: ["suggest", id],
    queryFn: () => suggestQuestions(id),
    enabled: !!id,
  });

  useEffect(() => {
    document.title = `${data?.filename ?? "Document"} • AI Legal Document Analyzer`;
    console.log("Received document data:", data); // Debugging log
    console.log("Risk score:", data?.risk_score); // Debugging log
    console.log("Lawyer recommendation:", data?.lawyer_recommendation, "Urgency:", data?.lawyer_urgency); // Debugging log
  }, [data]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chat]);

  const riskLevel = useMemo(() => {
    const score = Math.max(data?.risk_score ?? 0, 0.1); // Ensure minimum 0.1
    if (score <= 0.3) return { variant: "default" as const, color: "text-emerald-600", bg: "bg-emerald-50", label: "Low Risk" };
    if (score <= 0.7) return { variant: "secondary" as const, color: "text-amber-600", bg: "bg-amber-50", label: "Medium Risk" };
    return { variant: "destructive" as const, color: "text-red-600", bg: "bg-red-50", label: "High Risk" };
  }, [data?.risk_score]);

  const qMutation = useMutation({
    mutationFn: async () => {
      if (!question.trim()) throw new Error("Please enter a question");
      const timestamp = new Date().toISOString();
      setChat((c) => [...c, { role: "user", content: question, timestamp }]);
      const res = await askQuestion(id, question);
      return { ...res, timestamp };
    },
    onSuccess: (res) => {
      setChat((c) => [...c, { role: "assistant", content: res.answer, sections: res.relevant_sections, timestamp: res.timestamp }]);
      setQuestion("");
    },
    onError: (err: any) => {
      toast({ title: "Question Failed", description: err?.message ?? "Unexpected error", variant: "destructive" });
    },
  });

  const exportToPDF = () => {
    try {
      const doc = new jsPDF();
      const pageWidth = doc.internal.pageSize.getWidth();
      const margin = 15;
      const maxWidth = pageWidth - 2 * margin;
      let yOffset = margin;

      // Set fonts
      doc.setFont("helvetica", "bold");

      // Title
      doc.setFontSize(20);
      doc.text("Legal Document Analysis Report", pageWidth / 2, yOffset, { align: "center" });
      yOffset += 10;

      // Filename
      doc.setFontSize(16);
      doc.text(data?.filename || "Document", pageWidth / 2, yOffset, { align: "center" });
      yOffset += 10;

      // Add a horizontal line
      doc.setLineWidth(0.5);
      doc.line(margin, yOffset, pageWidth - margin, yOffset);
      yOffset += 10;

      // Risk Assessment Section
      doc.setFontSize(14);
      doc.setFont("helvetica", "bold");
      doc.text("Risk Assessment", margin, yOffset);
      yOffset += 7;

      doc.setFontSize(12);
      doc.setFont("helvetica", "normal");
      doc.text(`Risk Score: ${(Math.max(data?.risk_score ?? 0, 0.1) * 100).toFixed(0)}% (${riskLevel.label})`, margin + 5, yOffset);
      yOffset += 6;
      doc.text(`Confidence Score: ${(data?.confidence_score * 100).toFixed(0)}%`, margin + 5, yOffset);
      yOffset += 6;
      doc.text(
        `Legal Review: ${data?.lawyer_recommendation 
          ? `Recommended (${data?.lawyer_urgency.toUpperCase()} priority)` 
          : "Not required (Suitable for self-review)"
        }`,
        margin + 5, yOffset
      );
      

      // Executive Summary Section
      doc.setFontSize(14);
      doc.setFont("helvetica", "bold");
      doc.text("Executive Summary", margin, yOffset);
      yOffset += 7;

      doc.setFontSize(12);
      doc.setFont("helvetica", "normal");
      const summaryLines = doc.splitTextToSize(data?.summary || "No summary available.", maxWidth - 10);
      doc.text(summaryLines, margin + 5, yOffset);
      yOffset += summaryLines.length * 6 + 10;

      // Key Clauses Section
      doc.setFontSize(14);
      doc.setFont("helvetica", "bold");
      doc.text("Key Clauses", margin, yOffset);
      yOffset += 7;

      doc.setFontSize(12);
      doc.setFont("helvetica", "normal");
      (data?.key_clauses || []).forEach((clause, index) => {
        if (yOffset > doc.internal.pageSize.getHeight() - 20) {
          doc.addPage();
          yOffset = margin;
        }
        const clauseLines = doc.splitTextToSize(`${index + 1}. ${clause}`, maxWidth - 10);
        doc.text(clauseLines, margin + 5, yOffset);
        yOffset += clauseLines.length * 6 + 3;
      });
      yOffset += 5;

      // Risk Factors Section
      doc.setFontSize(14);
      doc.setFont("helvetica", "bold");
      doc.text("Risk Factors", margin, yOffset);
      yOffset += 7;

      doc.setFontSize(12);
      doc.setFont("helvetica", "normal");
      (data?.red_flags || []).forEach((flag, index) => {
        if (yOffset > doc.internal.pageSize.getHeight() - 20) {
          doc.addPage();
          yOffset = margin;
        }
        const flagLines = doc.splitTextToSize(`${index + 1}. ${flag}`, maxWidth - 10);
        doc.text(flagLines, margin + 5, yOffset);
        yOffset += flagLines.length * 6 + 3;
      });

      // Add footer with timestamp
      const timestamp = new Date().toLocaleString();
      doc.setFontSize(10);
      doc.setFont("helvetica", "italic");
      doc.text(`Generated on: ${timestamp}`, margin, doc.internal.pageSize.getHeight() - 10);

      // Download the PDF
      doc.save(`${data?.filename || "document"}-analysis-report.pdf`);
      toast({ title: "Success", description: "Report exported successfully as PDF" });
    } catch (err: any) {
      console.error("PDF export error:", err);
      toast({ title: "Export Failed", description: err?.message ?? "Unexpected error", variant: "destructive" });
    }
  };

  if (isLoading) {
    return (
      <div className="flex flex-col justify-center items-center h-screen">
        <motion.div
          initial={{ scale: 0.8, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ duration: 0.5 }}
          className="text-center"
        >
          <Loader2 className="h-12 w-12 animate-spin text-blue-600 mx-auto mb-4" />
          <p className="text-slate-600 text-lg font-medium">Loading document analysis...</p>
          <p className="text-slate-400 text-sm mt-2">Please wait while we process your document</p>
        </motion.div>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="flex flex-col justify-center items-center h-screen bg-gradient-to-br from-red-50 to-orange-50">
        <motion.div
          initial={{ scale: 0.8, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ duration: 0.5 }}
          className="text-center"
        >
          <AlertTriangle className="h-12 w-12 text-red-500 mx-auto mb-4" />
          <p className="text-red-600 text-lg font-medium">Failed to load document</p>
          <Button 
            onClick={() => navigate('/documents')} 
            className="mt-4"
            variant="outline"
          >
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back to Documents
          </Button>
        </motion.div>
      </div>
    );
  }

  return (
    <TooltipProvider>
      <div className="block">
        <motion.div
          className="container"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          {/* Header */}
          <motion.div 
            className="mb-8"
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
          >
            <Button 
              variant="ghost" 
              onClick={() => navigate('/documents')}
              className="mb-4 text-slate-600 hover:text-slate-900"
            >
              <ArrowLeft className="h-4 w-4 mr-2" />
              Back to Documents
            </Button>
            
            <div className="flex items-center justify-between">
              <div>
                <h1 className="text-3xl font-bold text-slate-900 mb-2">{data.filename}</h1>
                <p className="text-slate-600">AI-powered legal document analysis</p>
              </div>
              <Button
                onClick={exportToPDF}
                className="bg-blue-600 hover:bg-blue-700 text-white"
              >
                <Download className="h-4 w-4 mr-2" />
                Export Report
              </Button>
            </div>
          </motion.div>

          <div className="grid grid-cols-1 xl:grid-cols-3 gap-8">
            {/* Main Content */}
            <div className="xl:col-span-2 space-y-6">
              {/* Risk Overview */}
              <motion.div
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.3 }}
              >
                <Card className="border-0 shadow-xl bg-white/80 backdrop-blur-sm">
                  <CardHeader className="pb-4">
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-xl text-slate-900 flex items-center gap-2">
                        <Shield className="h-5 w-5 text-blue-600" />
                        Risk Assessment
                      </CardTitle>
                      <Badge variant={riskLevel.variant} className="text-sm font-semibold px-3 py-1">
                        {riskLevel.label}
                      </Badge>
                    </div>
                  </CardHeader>
                  <CardContent className="space-y-6">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div className={`p-4 rounded-xl ${riskLevel.bg} border border-slate-200`}>
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-sm font-medium text-slate-700">Risk Score</span>
                          <span className={`text-2xl font-bold ${riskLevel.color}`}>
                            {(Math.max(data.risk_score ?? 0, 0.1) * 100).toFixed(0)}%
                          </span>
                        </div>
                        <Progress 
                          value={Math.max(data.risk_score ?? 0, 0.1) * 100} 
                          className="h-2"
                        />
                      </div>
                      
                      <div className="p-4 rounded-xl bg-blue-50 border border-slate-200">
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-sm font-medium text-slate-700">Confidence</span>
                          <span className="text-2xl font-bold text-blue-600">
                            {(data.confidence_score * 100).toFixed(0)}%
                          </span>
                        </div>
                        <Progress 
                          value={data.confidence_score * 100} 
                          className="h-2"
                        />
                      </div>
                    </div>

                    {/* Lawyer Recommendation Section */}
                    <div className={`p-4 rounded-xl ${data.lawyer_recommendation ? 'bg-yellow-50 border-yellow-200' : 'bg-green-50 border-green-200'} border`}>
                      <div className="flex items-center gap-2 mb-2">
                        <TrendingUp className="h-4 w-4 text-slate-600" />
                        <span className="text-sm font-medium text-slate-700">Lawyer Review Recommendation</span>
                      </div>
                      <p className="text-sm font-semibold text-slate-900">
                        {data.lawyer_recommendation 
                          ? `Lawyer consultation recommended (${data.lawyer_urgency.toUpperCase()} priority)`
                          : "Document suitable for self-review"
                        }
                      </p>
                      
                    </div>
                  </CardContent>
                </Card>
              </motion.div>

              {/* Document Analysis */}
              <motion.div
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.4 }}
              >
                <Card className="border-0 shadow-xl bg-white/80 backdrop-blur-sm">
                  <CardHeader>
                    <CardTitle className="text-xl text-slate-900 flex items-center gap-2">
                      <FileText className="h-5 w-5 text-blue-600" />
                      Document Analysis
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <Tabs defaultValue="summary" className="w-full">
                      <TabsList className="grid w-full grid-cols-2 bg-slate-100 p-1 rounded-lg">
                        <TabsTrigger 
                          value="summary"
                          className="data-[state=active]:bg-white data-[state=active]:shadow-sm rounded-md"
                        >
                          Executive Summary
                        </TabsTrigger>
                        <TabsTrigger 
                          value="clauses"
                          className="data-[state=active]:bg-white data-[state=active]:shadow-sm rounded-md"
                        >
                          Key Points & Risks
                        </TabsTrigger>
                      </TabsList>
                      
                      <TabsContent value="summary" className="mt-6">
                        <div className="prose prose-slate max-w-none">
                          <div className="bg-slate-50 p-6 rounded-xl border border-slate-200">
                            <p className="text-slate-700 leading-relaxed whitespace-pre-wrap">
                              {data.summary}
                            </p>
                          </div>
                        </div>
                      </TabsContent>
                      
                      <TabsContent value="clauses" className="mt-6">
                        <div className="grid gap-6">
                          <div>
                            <h3 className="text-lg font-semibold text-slate-900 mb-4 flex items-center gap-2">
                              <Sparkles className="h-4 w-4 text-blue-600" />
                              Key Clauses
                            </h3>
                            <div className="space-y-3">
                              {data.key_clauses.map((clause, i) => (
                                <motion.div
                                  key={i}
                                  className="p-4 bg-blue-50 border border-blue-200 rounded-lg hover:bg-blue-100 transition-colors"
                                  initial={{ opacity: 0, y: 10 }}
                                  animate={{ opacity: 1, y: 0 }}
                                  transition={{ delay: i * 0.1 }}
                                >
                                  <p className="text-slate-700">{clause}</p>
                                </motion.div>
                              ))}
                            </div>
                          </div>
                          
                          <div>
                            <h3 className="text-lg font-semibold text-slate-900 mb-4 flex items-center gap-2">
                              <AlertTriangle className="h-4 w-4 text-red-600" />
                              Risk Factors
                            </h3>
                            <div className="space-y-3">
                              {data.red_flags.map((flag, i) => (
                                <motion.div
                                  key={i}
                                  className="p-4 bg-red-50 border border-red-200 rounded-lg hover:bg-red-100 transition-colors"
                                  initial={{ opacity: 0, y: 10 }}
                                  animate={{ opacity: 1, y: 0 }}
                                  transition={{ delay: i * 0.1 }}
                                >
                                  <div className="flex items-start gap-3">
                                    <AlertTriangle className="h-4 w-4 text-red-600 mt-0.5 flex-shrink-0" />
                                    <p className="text-slate-700">{flag}</p>
                                  </div>
                                </motion.div>
                              ))}
                            </div>
                          </div>
                        </div>
                      </TabsContent>
                    </Tabs>
                  </CardContent>
                </Card>
              </motion.div>
            </div>

            {/* Chat Sidebar */}
            <motion.div
              className="xl:col-span-1"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.5 }}
            >
              <div className="sticky top-6 space-y-6">
                {/* Suggested Questions */}
                <Card className="border-0 shadow-xl bg-white/80 backdrop-blur-sm">
                  <CardHeader className="pb-4">
                    <CardTitle className="text-lg text-slate-900 flex items-center gap-2">
                      <Sparkles className="h-4 w-4 text-blue-600" />
                      Quick Questions
                    </CardTitle>
                    <CardDescription>Click to ask instantly</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2">
                      {sugg?.questions?.slice(0, 4).map((q, i) => (
                        <motion.div
                          key={i}
                          initial={{ opacity: 0, y: 10 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: 0.6 + i * 0.1 }}
                        >
                          <Button
                            variant="ghost"
                            className="w-full text-left justify-start h-auto p-3 text-sm text-slate-700 hover:bg-blue-50 hover:text-blue-700 border border-slate-200 hover:border-blue-300 transition-all"
                            onClick={() => handleAsk(q)}
                          >
                            <span className="line-clamp-2">{q}</span>
                          </Button>
                        </motion.div>
                      )) ?? (
                        <p className="text-sm text-slate-500 text-center py-4">
                          No suggestions available
                        </p>
                      )}
                    </div>
                  </CardContent>
                </Card>

                {/* Chat Interface */}
                <Card className="border-0 shadow-xl bg-white/80 backdrop-blur-sm">
                  <CardHeader className="pb-4">
                    <CardTitle className="text-lg text-slate-900 flex items-center gap-2">
                      <MessageSquare className="h-4 w-4 text-blue-600" />
                      AI Legal Assistant
                    </CardTitle>
                    <CardDescription>Ask specific questions about your document</CardDescription>
                  </CardHeader>
                  
                  <CardContent className="p-0">
                    {/* Chat Messages */}
                    <div className="h-96 overflow-y-auto px-6 space-y-4">
                      {chat.length === 0 && (
                        <div className="flex flex-col items-center justify-center h-full text-center">
                          <Bot className="h-12 w-12 text-slate-300 mb-3" />
                          <p className="text-slate-500 text-sm">
                            Start a conversation by asking a question or selecting a suggestion above.
                          </p>
                        </div>
                      )}
                      
                      {chat.map((message, idx) => (
                        <motion.div
                          key={idx}
                          className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}
                          initial={{ opacity: 0, y: 20 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: idx * 0.05 }}
                        >
                          <div className={`max-w-[85%] p-3 rounded-lg ${
                            message.role === "user" 
                              ? "bg-blue-600 text-white" 
                              : "bg-slate-100 text-slate-800"
                          }`}>
                            <div className="flex items-center gap-2 mb-1">
                              {message.role === "assistant" && <Bot className="h-3 w-3" />}
                              <span className="text-xs opacity-75">
                                {message.role === "user" ? "You" : "AI Lawyer"}
                              </span>
                              {message.timestamp && (
                                <span className="text-xs opacity-50">
                                  {new Date(message.timestamp).toLocaleTimeString()}
                                </span>
                              )}
                            </div>
                            <p className="text-sm leading-relaxed whitespace-pre-wrap">
                              {message.content}
                            </p>
                            {message.sections && (
                              <div className="mt-2 pt-2 border-t border-slate-200 text-xs opacity-75">
                                <strong>Referenced sections:</strong> {message.sections.length}
                              </div>
                            )}
                          </div>
                        </motion.div>
                      ))}
                      
                      <AnimatePresence>
                        {qMutation.isPending && (
                          <motion.div
                            className="flex justify-start"
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: 20 }}
                          >
                            <div className="bg-slate-100 p-3 rounded-lg flex items-center gap-2">
                              <Bot className="h-4 w-4 text-blue-600 animate-pulse" />
                              <span className="text-sm text-slate-600">
                                AI is thinking...
                              </span>
                            </div>
                          </motion.div>
                        )}
                      </AnimatePresence>
                      
                      <div ref={bottomRef} />
                    </div>
                    
                    {/* Chat Input */}
                    <div className="p-4 border-t border-slate-200">
                      <div className="flex gap-2">
                        <Input
                          placeholder="Ask about your document..."
                          value={question}
                          onChange={(e) => setQuestion(e.target.value)}
                          onKeyDown={(e) => {
                            if (e.key === "Enter" && !qMutation.isPending && question.trim()) {
                              qMutation.mutate();
                            }
                          }}
                          className="border-slate-200 focus:border-blue-500 focus:ring-blue-500"
                          disabled={qMutation.isPending}
                        />
                        <Button
                          onClick={() => qMutation.mutate()}
                          disabled={qMutation.isPending || !question.trim()}
                          size="sm"
                          className="bg-blue-600 hover:bg-blue-700 px-3"
                        >
                          {qMutation.isPending ? (
                            <Loader2 className="h-4 w-4 animate-spin" />
                          ) : (
                            <Send className="h-4 w-4" />
                          )}
                        </Button>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </motion.div>
          </div>
        </motion.div>
      </div>
    </TooltipProvider>
  );
}