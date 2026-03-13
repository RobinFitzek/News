export function MetricDisplay({ label, value, unit = '', className = '' }) {
  return (
    <div className={`flex justify-between items-center ${className}`}>
      <span className="text-sm text-gray-500 font-mono">{label}</span>
      <span className="text-sm font-mono">{value}{unit}</span>
    </div>
  )
}