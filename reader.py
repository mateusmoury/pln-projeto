import xlrd

def read(path):
  posts_document = xlrd.open_workbook(path)
  posts = posts_document.sheet_by_index(0)
  annotated_texts = []
  for i in range(0, posts.nrows):
    annotated_texts.append((posts.cell(i,0).value, posts.cell(i, 1).value))
  return annotated_texts

if __name__ == '__main__':
  annotated_texts = read('blog-gender-dataset.xlsx')
  for text in annotated_texts:
    print(text[0])
    print(text[1])
