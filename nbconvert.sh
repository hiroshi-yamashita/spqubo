jupyter nbconvert \
  --to python \
  --TemplateExporter.exclude_input_prompt=True \
  --TemplateExporter.exclude_output_prompt=True \
  --output $2 \
  $1
sed -i '/^get_ipython/d' $2 
sed -i 's/^display/print/' $2 
