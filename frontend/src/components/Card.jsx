export function Card({ title, children, className = '' }) {
  return (
    <div className={`border border-gray-300 p-4 ${className}`}>
      {title && (
        <h3 className="text-lg font-serif mb-4 border-b border-gray-300 pb-2">
          {title}
        </h3>
      )}
      <div className="space-y-3">
        {children}
      </div>
    </div>
  )
}