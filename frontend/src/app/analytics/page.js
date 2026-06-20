'use client';

import { useState, useEffect } from 'react';
import { getAnalytics } from '@/lib/api';
import PageTransition from '@/components/PageTransition';
import MetricCard from '@/components/MetricCard';
import { IconChart, IconDatabase, IconActivity, IconShield, IconMapPin } from '@/components/SvgIcons';
import {
  BarChart, Bar, LineChart, Line, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts';

const COLORS = ['#ef4444', '#3b82f6', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899', '#14b8a6', '#f97316', '#6366f1', '#84cc16'];

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
        <p style={{ fontWeight: 600, marginBottom: 4, color: '#f0f0f5' }}>{label}</p>
        {payload.map((p, i) => (
          <p key={i} style={{ color: p.color || '#a0aec0' }}>
            {p.name}: <strong>{typeof p.value === 'number' ? p.value.toLocaleString() : p.value}</strong>
          </p>
        ))}
      </div>
    );
  }
  return null;
};

export default function AnalyticsPage() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getAnalytics()
      .then(setData)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <PageTransition><div className="loading-spinner" /></PageTransition>;
  if (!data) return <PageTransition><p style={{ color: 'var(--text-secondary)' }}>Failed to load analytics data.</p></PageTransition>;

  const { overview, cause_distribution, hourly_distribution, zone_distribution, priority_by_cause } = data;

  // Prepare stacked bar data
  const stackedData = {};
  priority_by_cause.forEach(item => {
    const cause = item.event_cause;
    if (!stackedData[cause]) stackedData[cause] = { cause };
    stackedData[cause][item.priority] = item.count;
  });
  const stackedBarData = Object.values(stackedData);

  return (
    <PageTransition>
      <div className="page-header">
        <h1 className="page-header__title">Dataset Analytics Dashboard</h1>
        <p className="page-header__desc">
          Exploratory insights from the raw historical ASTRAM congestion database.
        </p>
      </div>

      {/* KPI Cards */}
      <div className="metrics-grid" style={{ gridTemplateColumns: 'repeat(4, 1fr)' }}>
        <MetricCard
          icon={<IconDatabase size={18} />}
          title="Total Events"
          value={overview.total_events}
          delay={0}
        />
        <MetricCard
          icon={<IconActivity size={18} />}
          title="Unplanned Events"
          value={overview.unplanned}
          delay={100}
        />
        <MetricCard
          icon={<IconShield size={18} />}
          title="High Priority"
          value={overview.high_priority}
          type="danger"
          delay={200}
        />
        <MetricCard
          icon={<IconChart size={18} />}
          title="Road Closures"
          value={overview.road_closures}
          type="warning"
          delay={300}
        />
      </div>

      {/* Charts Grid */}
      <div className="grid-2">
        {/* Cause Distribution */}
        <div className="chart-card" style={{ animationDelay: '0.1s' }}>
          <div className="chart-card__title">
            <IconChart size={18} />
            Distribution of Events by Cause
          </div>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={cause_distribution} margin={{ top: 5, right: 10, left: 0, bottom: 60 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="cause" angle={-45} textAnchor="end" height={80} tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip content={<CustomTooltip />} />
              <defs>
                <linearGradient id="barGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#ef4444" />
                  <stop offset="100%" stopColor="#991b1b" />
                </linearGradient>
              </defs>
              <Bar dataKey="count" fill="url(#barGrad)" radius={[4, 4, 0, 0]} animationDuration={1200} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Hourly Distribution */}
        <div className="chart-card" style={{ animationDelay: '0.2s' }}>
          <div className="chart-card__title">
            <IconActivity size={18} />
            Congestion Peak Hour Distribution
          </div>
          <ResponsiveContainer width="100%" height={320}>
            <LineChart data={hourly_distribution} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="hour" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip content={<CustomTooltip />} />
              <Line
                type="monotone"
                dataKey="count"
                stroke="#ef4444"
                strokeWidth={2.5}
                dot={{ r: 4, fill: '#fff', strokeWidth: 2, stroke: '#ef4444' }}
                activeDot={{ r: 6 }}
                animationDuration={1500}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Zone Pie */}
        <div className="chart-card" style={{ animationDelay: '0.3s' }}>
          <div className="chart-card__title">
            <IconMapPin size={18} />
            Share of Events by Zone
          </div>
          <ResponsiveContainer width="100%" height={320}>
            <PieChart>
              <Pie
                data={zone_distribution}
                dataKey="count"
                nameKey="zone"
                cx="50%"
                cy="50%"
                innerRadius={70}
                outerRadius={120}
                paddingAngle={2}
                animationDuration={1200}
              >
                {zone_distribution.map((_, i) => (
                  <Cell key={i} fill={COLORS[i % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip content={<CustomTooltip />} />
              <Legend wrapperStyle={{ fontSize: '0.78rem', fontFamily: 'Outfit' }} />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Stacked Priority by Cause */}
        <div className="chart-card" style={{ animationDelay: '0.4s' }}>
          <div className="chart-card__title">
            <IconShield size={18} />
            Priority Breakdown by Cause
          </div>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={stackedBarData} margin={{ top: 5, right: 10, left: 0, bottom: 60 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="cause" angle={-45} textAnchor="end" height={80} tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip content={<CustomTooltip />} />
              <Legend wrapperStyle={{ fontSize: '0.78rem', fontFamily: 'Outfit' }} />
              <Bar dataKey="High" stackId="a" fill="#ef4444" radius={[4, 4, 0, 0]} animationDuration={1200} />
              <Bar dataKey="Low" stackId="a" fill="#10b981" animationDuration={1200} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </PageTransition>
  );
}
