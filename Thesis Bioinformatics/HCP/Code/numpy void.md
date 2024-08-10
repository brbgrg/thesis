When you encounter a `numpy.void` type in this context, it typically means you're dealing with a structured NumPy array, often resulting from loading MATLAB structs. The `numpy.void` type can contain multiple fields, similar to attributes or keys in a dictionary, each potentially holding an array or other data structure.

Given that [`sc_content`](vscode-file://vscode-app/c:/Users/barbo/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.html "GNN.py") is of type `numpy.void`, you can access its fields similarly to how you would access elements in a dictionary or attributes of an object. However, the exact names of these fields depend on the original MATLAB struct's design. Assuming you want to extract three arrays that might represent different age groups or conditions (e.g., young, adult, old), you first need to know the names of these fields.

Here's a general approach to inspect and extract the arrays:

1. **Inspect Available Fields**: Use the `dtype.names` attribute to list all field names contained within the `numpy.void` object.
    
    print(sc_content.dtype.names)
    
2. **Extract Arrays Using Field Names**: Once you know the field names, you can extract the arrays by accessing these fields directly.
    
    Assuming the fields are named `'young'`, `'adult'`, and `'old'` (replace these with the actual field names you find), here's how you would extract the arrays:
    
    young_matrix = sc_content['young']
    
    adult_matrix = sc_content['adult']
    
    old_matrix = sc_content['old']
    
3. **Inspect Extracted Arrays**: It's a good practice to inspect the extracted arrays to confirm their types and shapes are as expected.
    
    print("Young Matrix:", type(young_matrix), young_matrix.shape)
    
    print("Adult Matrix:", type(adult_matrix), adult_matrix.shape)
    
    print("Old Matrix:", type(old_matrix), old_matrix.shape)
    

This approach allows you to navigate the structured data loaded from MATLAB files and extract the specific arrays you're interested in.