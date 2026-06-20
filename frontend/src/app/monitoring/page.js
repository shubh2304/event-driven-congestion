'use client';

import { useState, useEffect } from 'react';
import { getMonitoringPredictions, getDrift, getRetrainStatus } from '@/lib/api';
import PageTransition from '@/components/PageTransition';
import MetricCard from '@/components/MetricCard';
import { IconMonitor, IconShield, IconRoadblock, IconCheck, IconWarning, IconAlert, IconRefresh, IconChart, IconDatabase, IconActivity, IconPerformance } from '@/components/SvgIcons';
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
        <p style={{ color: '#a0aec0' }}>Count: <strong>{payload[0]?.value}</strong></p>
      </div>
    );
  }
  return null;
};

function buildHistogramData(values, bins = 20) {
  if (!values || values.length === 0) return [];
  const min = Math.min(...values);
  const max = Math.max(...values);
  const binWidth = (max - min) / bins || 1;
  const histogram = Array.from({ length: bins }, (_, i) => ({
    range: `${(min + i * binWidth).toFixed(0)}-${(min + (i + 1) * binWidth).toFixed(0)}`,
    count: 0
  }));
  values.forEach(v => {
    const idx = Math.min(Math.floor((v - min) / binWidth), bins - 1);
    if (histogram[idx]) histogram[idx].count++;
  });
  return histogram;
}

export default function MonitoringPage() {
  const [predictions, setPredictions] = useState(null);
  const [drift, setDrift] = useState(null);
  const [retrain, setRetrain] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      getMonitoringPredictions().catch(() => null),
      getDrift().catch(() => null),
      getRetrainStatus().catch(() => null),
    ]).then(([pred, dr, rt]) => {
      setPredictions(pred);
      setDrift(dr);
      setRetrain(rt);
    }).finally(() => setLoading(false));
  }, []);

  if (loading) return <PageTransition><div className="loading-spinner" /></PageTransition>;

  const hasPredData = predictions?.has_data;

  const prioHistogram = hasPredData ? buildHistogramData(predictions.priority_values) : [];
  const closeHistogram = hasPredData ? buildHistogramData(predictions.closure_values) : [];

  // Parse drift results
  const driftRows = [];
  if (drift?.prediction_drift) {
    Object.entries(drift.prediction_drift).forEach(([col, info]) => {
      if (info && typeof info === 'object' && 'psi' in info) {
        driftRows.push({
          metric: col.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
          psi: info.psi,
          status: info.status
        });
      }
    });
  }

  const getStatusIcon = (status) => {
    if (!status) return null;
    const s = status.toLowerCase();
    if (s.includes('stable') || s.includes('no')) return <><span className="status-dot status-dot--stable" /><span className="status-stable">Stable</span></>;
    if (s.includes('moderate') || s.includes('drift')) return <><span className="status-dot status-dot--drifting" /><span className="status-drifting">Drifting</span></>;
    return <><span className="status-dot status-dot--critical" /><span className="status-critical">Significant Shift</span></>;
  };

  return (
    <PageTransition>
      <div className="page-header">
        <h1 className="page-header__title">Model Monitoring & Drift Detection</h1>
        <p className="page-header__desc">
          Track prediction quality, detect data drift, and monitor model health over time.
        </p>
      </div>

      {!hasPredData ? (
        <div className="empty-state">
          <IconMonitor size={48} />
          <div className="empty-state__title">No prediction logs yet</div>
          <div className="empty-state__desc">
            Use the Event Impact Predictor or ASTRAM Assistant to generate predictions, and they will appear here.
          </div>
        </div>
      ) : (
        <>
          {/* Prediction Volume */}
          <h2 className="section-title">
            <IconChart size={20} />
            Prediction Volume
          </h2>
          <div className="metrics-grid" style={{ gridTemplateColumns: 'repeat(4, 1fr)' }}>
            <MetricCard icon={<IconDatabase size={18} />} title="Total Predictions" value={predictions.total} delay={0} />
            <MetricCard icon={<IconActivity size={18} />} title="Last 24 Hours" value={predictions.last_24h} delay={100} />
            <MetricCard icon={<IconShield size={18} />} title="Avg Priority Risk" value={`${predictions.avg_priority_risk}%`} delay={200} />
            <MetricCard icon={<IconRoadblock size={18} />} title="Avg Closure Risk" value={`${predictions.avg_closure_risk}%`} delay={300} />
          </div>

          {/* Distribution Histograms */}
          <h2 className="section-title">
            <IconChart size={20} />
            Prediction Distributions
          </h2>
          <div className="grid-2">
            <div className="chart-card">
              <div className="chart-card__title">Priority Risk Distribution (%)</div>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={prioHistogram}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="range" tick={{ fontSize: 10 }} />
                  <YAxis tick={{ fontSize: 11 }} />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="count" fill="#ef4444" radius={[3, 3, 0, 0]} animationDuration={1000} />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="chart-card">
              <div className="chart-card__title">Closure Risk Distribution (%)</div>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={closeHistogram}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="range" tick={{ fontSize: 10 }} />
                  <YAxis tick={{ fontSize: 11 }} />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="count" fill="#3b82f6" radius={[3, 3, 0, 0]} animationDuration={1000} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Drift Detection */}
          <h2 className="section-title">
            <IconWarning size={20} />
            Drift Detection (PSI)
          </h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginBottom: '1rem' }}>
            PSI measures how much prediction distributions have shifted.
            Values below 0.10 are stable, 0.10 to 0.25 indicate moderate drift,
            above 0.25 suggests significant drift requiring retraining.
          </p>

          {driftRows.length > 0 ? (
            <div className="card" style={{ overflow: 'auto', marginBottom: '1.5rem' }}>
              <table className="data-table" id="drift-table">
                <thead>
                  <tr>
                    <th>Metric</th>
                    <th>PSI Value</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {driftRows.map((row, i) => (
                    <tr key={i}>
                      <td>{row.metric}</td>
                      <td>{typeof row.psi === 'number' ? row.psi.toFixed(4) : row.psi}</td>
                      <td>{getStatusIcon(row.status)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="card" style={{ padding: '1.5rem', color: 'var(--text-secondary)', marginBottom: '1.5rem' }}>
              <IconWarning size={16} style={{ marginRight: 6 }} />
              {predictions.total < 40
                ? `Need at least 40 predictions for drift detection. Current count: ${predictions.total}.`
                : 'Drift analysis returned no results.'
              }
            </div>
          )}

          {/* Recent Prediction Log */}
          <h2 className="section-title">
            <IconPerformance size={20} />
            Recent Prediction Log
          </h2>
          <div className="card" style={{ overflow: 'auto', marginBottom: '1.5rem' }}>
            <table className="data-table" id="prediction-log-table">
              <thead>
                <tr>
                  <th>Timestamp</th>
                  <th>Event Cause</th>
                  <th>Zone</th>
                  <th>Priority Prob</th>
                  <th>Closure Prob</th>
                  <th>Duration (min)</th>
                </tr>
              </thead>
              <tbody>
                {predictions.recent.map((row, i) => (
                  <tr key={i}>
                    <td>{row.timestamp ? new Date(row.timestamp).toLocaleString() : '-'}</td>
                    <td>{row.event_cause || '-'}</td>
                    <td>{row.zone || '-'}</td>
                    <td>{row.priority_prob != null ? (parseFloat(row.priority_prob) * 100).toFixed(1) + '%' : '-'}</td>
                    <td>{row.closure_prob != null ? (parseFloat(row.closure_prob) * 100).toFixed(1) + '%' : '-'}</td>
                    <td>{row.duration_est_min != null ? Math.round(row.duration_est_min) : '-'}</td>
                  </tr>
                ))}
                {predictions.recent.length === 0 && (
                  <tr><td colSpan="6" style={{ textAlign: 'center', color: 'var(--text-tertiary)' }}>No recent predictions</td></tr>
                )}
              </tbody>
            </table>
          </div>
        </>
      )}

      {/* Online Learning Status */}
      <h2 className="section-title">
        <IconRefresh size={20} />
        Online Learning Status
      </h2>
      {retrain?.has_retrained ? (
        <div className="grid-3">
          <MetricCard icon={<IconRefresh size={18} />} title="Last Retrain" value={retrain.last_retrain} />
          <MetricCard icon={<IconCheck size={18} />} title="Status" value={retrain.status} type={retrain.status === 'READY' ? 'success' : 'warning'} />
          <MetricCard icon={<IconDatabase size={18} />} title="Data Rows" value={typeof retrain.data_rows === 'number' ? retrain.data_rows : retrain.data_rows} />
        </div>
      ) : (
        <div className="card" style={{ padding: '1.5rem', color: 'var(--text-secondary)' }}>
          No retraining has been performed yet. Run the incremental retraining script:
          <pre style={{ marginTop: '0.75rem', background: 'var(--bg-primary)', padding: '0.75rem', borderRadius: 'var(--radius-sm)', fontSize: '0.85rem' }}>
            python online_update.py --retrain --window-months 6
          </pre>
        </div>
      )}
    </PageTransition>
  );
}
