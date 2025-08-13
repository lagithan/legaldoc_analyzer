import React, { useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { motion } from 'framer-motion';
import { useToast } from '@/hooks/use-toast';
import { BarChart, PieChart } from 'lucide-react';
import { getAnalytics } from '@/lib/api';

export default function AnalyticsDashboard() {
  const { toast } = useToast();

  const { data, isLoading, error } = useQuery({
    queryKey: ['analytics'],
    queryFn: getAnalytics,
  });

  useEffect(() => {
    document.title = 'Analytics • AI Legal Document Analyzer';
    console.log('Analytics data:', data); // Debugging log
    if (error) {
      toast({
        title: 'Analytics Failed',
        description: error?.message ?? 'Unexpected error',
        variant: 'destructive',
      });
    }
  }, [data, error, toast]);

  if (isLoading) {
    return (
      <div className="flex justify-center items-center h-screen">
        <motion.div
          initial={{ scale: 0.8, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ duration: 0.5 }}
          className="text-center"
        >
          <BarChart className="h-12 w-12 animate-spin text-blue-600 mx-auto mb-4" />
          <p className="text-slate-600 text-lg font-medium">Loading analytics...</p>
        </motion.div>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="flex justify-center items-center h-screen bg-gradient-to-br from-red-50 to-orange-50">
        <motion.div
          initial={{ scale: 0.8, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ duration: 0.5 }}
          className="text-center"
        >
          <BarChart className="h-12 w-12 text-red-500 mx-auto mb-4" />
          <p className="text-red-600 text-lg font-medium">Failed to load analytics</p>
        </motion.div>
      </div>
    );
  }

  return (
    <motion.div
      className="container py-8"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
    >
      <h1 className="text-3xl font-bold text-slate-900 mb-8">Document Analytics</h1>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {/* Total Documents */}
        <Card className="border-0 shadow-xl bg-white/80 backdrop-blur-sm">
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2">
              <BarChart className="h-5 w-5 text-blue-600" />
              Total Documents
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold text-slate-900">{data.total_documents}</p>
            <p className="text-sm text-slate-600 mt-1">
              Documents processed by the system
            </p>
          </CardContent>
        </Card>

        {/* Document Types */}
        <Card className="border-0 shadow-xl bg-white/80 backdrop-blur-sm">
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2">
              <PieChart className="h-5 w-5 text-blue-600" />
              Document Types
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {Object.entries(data.document_types).map(([type, count], i) => (
                <motion.div
                  key={type}
                  className="flex justify-between items-center"
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.1 }}
                >
                  <span className="text-sm text-slate-700 capitalize">
                    {type.replace('_', ' ')}
                  </span>
                  <Badge variant="secondary">{count}</Badge>
                </motion.div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Risk Distribution */}
        <Card className="border-0 shadow-xl bg-white/80 backdrop-blur-sm">
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2">
              <BarChart className="h-5 w-5 text-blue-600" />
              Risk Distribution
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {Object.entries(data.risk_distribution).map(([level, count], i) => (
                <motion.div
                  key={level}
                  className="flex justify-between items-center"
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.1 }}
                >
                  <span className="text-sm text-slate-700 capitalize">{level}</span>
                  <Badge variant={level === 'urgent' || level === 'high' ? 'destructive' : 'default'}>
                    {count}
                  </Badge>
                </motion.div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Average Confidence */}
        <Card className="border-0 shadow-xl bg-white/80 backdrop-blur-sm">
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2">
              <BarChart className="h-5 w-5 text-blue-600" />
              Average Confidence
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-slate-700">Confidence Score</span>
              <span className="text-xl font-bold text-blue-600">
                {(data.avg_confidence * 100).toFixed(0)}%
              </span>
            </div>
            <Progress value={data.avg_confidence * 100} className="h-2" />
          </CardContent>
        </Card>

        {/* Lawyer Recommendations */}
        <Card className="border-0 shadow-xl bg-white/80 backdrop-blur-sm">
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2">
              <BarChart className="h-5 w-5 text-blue-600" />
              Lawyer Recommendations
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold text-slate-900">
              {data.total_requiring_lawyer}
            </p>
            <p className="text-sm text-slate-600 mt-1">
              Documents requiring legal review
            </p>
          </CardContent>
        </Card>
      </div>
    </motion.div>
  );
}