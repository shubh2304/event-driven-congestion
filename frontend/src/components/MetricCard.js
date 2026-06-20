'use client';

import AnimatedCounter from './AnimatedCounter';

export default function MetricCard({ icon, title, value, subtitle, badge, delay = 0, type = 'default' }) {
  const colorMap = {
    default: '',
    danger: 'metric-card--danger',
    success: 'metric-card--success',
    warning: 'metric-card--warning',
    info: 'metric-card--info',
  };

  return (
    <div
      className={`metric-card ${colorMap[type] || ''}`}
      style={{ animationDelay: `${delay}ms` }}
    >
      <div className="metric-card__header">
        {icon && <span className="metric-card__icon">{icon}</span>}
        <span className="metric-card__title">{title}</span>
      </div>
      <div className="metric-card__value">
        {typeof value === 'number' ? (
          <AnimatedCounter
            value={value}
            decimals={value % 1 !== 0 ? 1 : 0}
            separator=","
          />
        ) : (
          value
        )}
      </div>
      {subtitle && <div className="metric-card__subtitle">{subtitle}</div>}
      {badge && (
        <div className={`badge ${badge.type === 'high' ? 'badge--danger' : 'badge--success'}`}>
          {badge.label}
        </div>
      )}
    </div>
  );
}
