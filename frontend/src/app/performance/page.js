'use client';

import { useState, useEffect } from 'react';
import { getMetrics } from '@/lib/api';
import PageTransition from '@/components/PageTransition';
import MetricCard from '@/components/MetricCard';
import { IconPerformance, IconShield, IconRoadblock, IconClock, IconChart } from '@/components/SvgIcons';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts';

const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <div style={{
        background: '#13131d',
        border: '1px solid rgba(255,255,255,0.1)',
        borderRadius: 8,
        padding: '0.6rem 0.85rem',
        fontSize: '0.82rem',
        fontFamily: 'Outfit, sans-serif',
      }}>
        <p style={{ fontWeight: 600, color: '#f0f0f5' }}>{label}</p>
        <p style={{ color: '#a0aec0' }}>Importance: <strong>{payload[0]?.value?.toLocaleString()}</strong></p>
      </div>
    );
  }
  return null;
};

export default function PerformancePage() {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState(0);

  useEffect(() => {
    getMetrics()
      .then(setMetrics)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <PageTransition><div className="loading-spinner" /></PageTransition>;
  if (!metrics) return <PageTransition><p style={{ color: 'var(--text-secondary)' }}>Failed to load metrics.</p></PageTransition>;

  const featureData = [...(metrics.priority_feature_importance || [])]
    .sort((a, b) => a.importance - b.importance);

  const tabs = [
    {
      label: 'Priority Model',
      icon: <IconShield size={16} />,
      content: (
        <div style={{ animation: 'fadeInUp 0.3s ease' }}>
          <h3 style={{ marginBottom: '0.5rem', fontSize: '1.1rem' }}>Priority Classification (LightGBM)</h3>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginBottom: '1.25rem' }}>
            The Priority classifier was trained using target encoding and SMOTE over-sampling.
            The model is tuned for macro-averaged F1 and general AUC robustness.
          </p>
          <div className="grid-2">
            <MetricCard icon={<IconShield size={18} />} title="ROC-AUC" value={String(metrics.priority_roc_auc ?? 'N/A')} type="success" />
            <MetricCard icon={<IconPerformance size={18} />} title="Weighted F1" value={String(metrics.priority_f1_weighted ?? 'N/A')} type="info" />
          </div>
        </div>
      )
    },
    {
      label: 'Road Closure Model',
      icon: <IconRoadblock size={16} />,
      content: (
        <div style={{ animation: 'fadeInUp 0.3s ease' }}>
          <h3 style={{ marginBottom: '0.5rem', fontSize: '1.1rem' }}>Road Closure Classification (XGBoost)</h3>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginBottom: '1.25rem' }}>
            The Road Closure model handles severe class imbalance. SMOTE is used,
            combined with threshold-tuning on validation predictions to maximize F1-score.
          </p>
          <div className="grid-2">
            <MetricCard icon={<IconRoadblock size={18} />} title="ROC-AUC" value={String(metrics.closure_roc_auc ?? 'N/A')} type="warning" />
            <MetricCard icon={<IconPerformance size={18} />} title="Optimal Threshold" value={String(metrics.closure_best_threshold ?? 'N/A')} type="info" />
          </div>
        </div>
      )
    },
    {
      label: 'Duration Regressor',
      icon: <IconClock size={16} />,
      content: (
        <div style={{ animation: 'fadeInUp 0.3s ease' }}>
          <h3 style={{ marginBottom: '0.5rem', fontSize: '1.1rem' }}>Duration Regression (LightGBM)</h3>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginBottom: '1.25rem' }}>
            The Duration Regressor predicts the elapsed time of a congestion incident in minutes.
            It is trained only on records containing a valid end time.
          </p>
          <div className="grid-3">
            <MetricCard icon={<IconClock size={18} />} title="MAE" value={`${metrics.duration_mae ?? 'N/A'} min`} type="danger" />
            <MetricCard icon={<IconClock size={18} />} title="Median AE" value={`${metrics.duration_medae ?? 'N/A'} min`} type="warning" />
            <MetricCard icon={<IconPerformance size={18} />} title="R-squared Score" value={String(metrics.duration_r2 ?? 'N/A')} type="info" />
          </div>
          <p style={{ color: 'var(--text-tertiary)', fontSize: '0.82rem', marginTop: '1rem', fontStyle: 'italic' }}>
            Note: The target duration contains high missing values in ASTRAM.
            Predictions are evaluated on back-transformed original values (minutes).
          </p>
        </div>
      )
    }
  ];

  return (
    <PageTransition>
      <div className="page-header">
        <h1 className="page-header__title">Model Performance & Validation Report</h1>
        <p className="page-header__desc">
          Post-training classification reports, regression stability scores, and predictor feature importance weights.
        </p>
      </div>

      {/* Tabs */}
      <div className="tabs">
        {tabs.map((tab, i) => (
          <button
            key={i}
            className={`tab ${activeTab === i ? 'tab--active' : ''}`}
            onClick={() => setActiveTab(i)}
            id={`model-tab-${i}`}
          >
            {tab.icon}
            <span style={{ marginLeft: 6 }}>{tab.label}</span>
          </button>
        ))}
      </div>

      <div className="card" style={{ marginBottom: '2rem' }}>
        {tabs[activeTab].content}
      </div>

      {/* Feature Importance */}
      <h2 className="section-title">
        <IconChart size={20} />
        Top Predictor Features (Feature Importance)
      </h2>

      <div className="chart-card">
        <ResponsiveContainer width="100%" height={400}>
          <BarChart data={featureData} layout="vertical" margin={{ top: 5, right: 30, left: 100, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" tick={{ fontSize: 11 }} />
            <YAxis type="category" dataKey="feature" tick={{ fontSize: 12 }} width={90} />
            <Tooltip content={<CustomTooltip />} />
            <defs>
              <linearGradient id="fiGrad" x1="0" y1="0" x2="1" y2="0">
                <stop offset="0%" stopColor="#991b1b" />
                <stop offset="100%" stopColor="#ef4444" />
              </linearGradient>
            </defs>
            <Bar dataKey="importance" fill="url(#fiGrad)" radius={[0, 4, 4, 0]} animationDuration={1500} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </PageTransition>
  );
}
