function Math(elem)
  -- Return the math content wrapped in $$ for display math
  return pandoc.RawInline('html', '$$' .. elem.text .. '$$')
end

function InlineMath(elem)
  -- Return the math content wrapped in $ for inline math
  return pandoc.RawInline('html', '$' .. elem.text .. '$')
end

