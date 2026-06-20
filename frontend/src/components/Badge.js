export default function Badge({ label, type = 'success' }) {
  const classMap = {
    success: 'badge--success',
    danger: 'badge--danger',
    warning: 'badge--warning',
    info: 'badge--info',
    neutral: 'badge--neutral',
  };

  return (
    <span className={`badge ${classMap[type] || classMap.neutral}`}>
      {label}
    </span>
  );
}
