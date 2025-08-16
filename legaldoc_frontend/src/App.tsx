import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Layout from './components/layout/Layout';
import Dashboard from './pages/Dashboard';
import Documents from './pages/Documents';
import DocumentView from './pages/DocumentView';
import Analytics from './pages/Analytics';
import { ROUTES } from './utils/constants';
import './styles/globals.css';

const App: React.FC = () => {
  return (
    <Router>
      <div className="App">
        <Routes>
          <Route path="/" element={<Layout />}>
            {/* Dashboard */}
            <Route index element={<Dashboard />} />

            {/* Documents */}
            <Route path="documents" element={<Documents />} />
            <Route path="documents/:id" element={<DocumentView />} />

            {/* Analytics */}
            <Route path="analytics" element={<Analytics />} />

            {/* Redirect any unknown routes to dashboard */}
            <Route path="*" element={<Navigate to="/" replace />} />
          </Route>
        </Routes>
      </div>
    </Router>
  );
};

export default App;