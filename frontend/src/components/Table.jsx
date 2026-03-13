export function Table({ headers, data, className = '' }) {
  return (
    <div className={`overflow-x-auto ${className}`}>
      <table className="w-full border-collapse">
        <thead>
          <tr className="border-b border-gray-300">
            {headers.map((header, index) => (
              <th key={index} className="text-left p-2 font-mono text-sm">
                {header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.map((row, rowIndex) => (
            <tr key={rowIndex} className="border-b border-gray-300">
              {row.map((cell, cellIndex) => (
                <td key={cellIndex} className="p-2 text-sm font-mono">
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}