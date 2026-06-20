'use client';

export default function GaugeChart({ value = 0, max = 100, size = 220, strokeWidth = 18 }) {
  const radius = (size - strokeWidth) / 2;
  const circumference = Math.PI * radius;
  const percentage = Math.min(Math.max(value / max, 0), 1);

  const getColor = (pct) => {
    if (pct < 0.4) return '#10b981';
    if (pct < 0.7) return '#f59e0b';
    return '#ef4444';
  };

  const getLabel = (pct) => {
    if (pct < 0.4) return 'LOW';
    if (pct < 0.7) return 'MEDIUM';
    return 'HIGH';
  };

  return (
    <div className="gauge-chart" style={{ width: size, height: size / 2 + 40 }}>
      <svg
        width={size}
        height={size / 2 + 20}
        viewBox={`0 0 ${size} ${size / 2 + 20}`}
      >
        {/* Background track segments */}
        <path
          d={`M ${strokeWidth / 2} ${size / 2} A ${radius} ${radius} 0 0 1 ${size - strokeWidth / 2} ${size / 2}`}
          fill="none"
          stroke="rgba(255,255,255,0.06)"
          strokeWidth={strokeWidth}
          strokeLinecap="round"
        />

        {/* Color segments */}
        <path
          d={`M ${strokeWidth / 2} ${size / 2} A ${radius} ${radius} 0 0 1 ${size - strokeWidth / 2} ${size / 2}`}
          fill="none"
          stroke="rgba(16, 185, 129, 0.12)"
          strokeWidth={strokeWidth}
          strokeDasharray={`${circumference * 0.4} ${circumference}`}
          strokeLinecap="round"
        />
        <path
          d={`M ${strokeWidth / 2} ${size / 2} A ${radius} ${radius} 0 0 1 ${size - strokeWidth / 2} ${size / 2}`}
          fill="none"
          stroke="rgba(245, 158, 11, 0.12)"
          strokeWidth={strokeWidth}
          strokeDasharray={`${circumference * 0.7} ${circumference}`}
          strokeDashoffset={`-${circumference * 0.4}`}
        />
        <path
          d={`M ${strokeWidth / 2} ${size / 2} A ${radius} ${radius} 0 0 1 ${size - strokeWidth / 2} ${size / 2}`}
          fill="none"
          stroke="rgba(239, 68, 68, 0.12)"
          strokeWidth={strokeWidth}
          strokeDasharray={`${circumference * 0.3} ${circumference}`}
          strokeDashoffset={`-${circumference * 0.7}`}
        />

        {/* Active arc */}
        <path
          d={`M ${strokeWidth / 2} ${size / 2} A ${radius} ${radius} 0 0 1 ${size - strokeWidth / 2} ${size / 2}`}
          fill="none"
          stroke={getColor(percentage)}
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeDasharray={`${circumference * percentage} ${circumference}`}
          className="gauge-chart__arc"
          style={{
            filter: `drop-shadow(0 0 8px ${getColor(percentage)}40)`,
          }}
        />

        {/* Center text */}
        <text
          x={size / 2}
          y={size / 2 - 10}
          textAnchor="middle"
          fill="white"
          fontSize="32"
          fontWeight="700"
          fontFamily="Outfit, sans-serif"
        >
          {value.toFixed(1)}%
        </text>
        <text
          x={size / 2}
          y={size / 2 + 16}
          textAnchor="middle"
          fill={getColor(percentage)}
          fontSize="14"
          fontWeight="600"
          fontFamily="Outfit, sans-serif"
          letterSpacing="2"
        >
          {getLabel(percentage)}
        </text>
      </svg>
      <div className="gauge-chart__label">Priority Risk Score</div>
    </div>
  );
}
