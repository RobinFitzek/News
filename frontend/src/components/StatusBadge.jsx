export function StatusBadge({ status, className = '' }) {
  const getStatusColor = () => {
    switch (status.toLowerCase()) {
      case 'active':
      case 'running':
      case 'success':
        return 'bg-positive text-black'
      case 'warning':
      case 'pending':
        return 'bg-gray-500 text-white'
      case 'error':
      case 'stopped':
      case 'danger':
        return 'bg-negative text-white'
      default:
        return 'bg-gray-500 text-white'
    }
  }

  return (
    <span className={`px-2 py-1 text-xs font-mono ${getStatusColor()} ${className}`}>
      {status.toUpperCase()}
    </span>
  )
}