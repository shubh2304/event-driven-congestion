'use client';

import { useEffect, useRef, useState } from 'react';

export default function AnimatedCounter({ 
  value, 
  duration = 1200, 
  decimals = 0, 
  prefix = '', 
  suffix = '',
  separator = ','
}) {
  const [displayValue, setDisplayValue] = useState(0);
  const startTime = useRef(null);
  const rafId = useRef(null);

  useEffect(() => {
    const target = typeof value === 'number' ? value : parseFloat(value) || 0;
    startTime.current = performance.now();

    function animate(now) {
      const elapsed = now - startTime.current;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      setDisplayValue(eased * target);
      if (progress < 1) {
        rafId.current = requestAnimationFrame(animate);
      }
    }

    rafId.current = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(rafId.current);
  }, [value, duration]);

  const formatted = displayValue.toFixed(decimals);
  const parts = formatted.split('.');
  if (separator) {
    parts[0] = parts[0].replace(/\B(?=(\d{3})+(?!\d))/g, separator);
  }

  return (
    <span className="animated-counter">
      {prefix}{parts.join('.')}{suffix}
    </span>
  );
}
