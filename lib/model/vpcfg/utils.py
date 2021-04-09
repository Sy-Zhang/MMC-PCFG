import numpy as np

def extract_parse(ispan, length, inc=1):
    tree = [(i, str(i)) for i in range(length)]
    tree = dict(tree)
    spans = []
    lprobs = []
    cover = ispan.nonzero(as_tuple=False)
    for i in range(cover.shape[0]):
        w, r, A = cover[i].tolist()
        a, b, c = w, r, A
        w = w + inc
        r = r + w
        l = r - w
        spans.append((l, r, A))
        lprobs.append(ispan[a, b, c])
        if l != r:
            span = '({} {})'.format(tree[l], tree[r])
            tree[r] = tree[l] = span
    return spans, tree[0], lprobs

def extract_parses(matrix, lengths, kbest=False, inc=1):
    batch = matrix.shape[1] if kbest else matrix.shape[0]
    spans = []
    trees = []
    lprobs = []
    for b in range(batch):
        if kbest:
            span, tree, _ = extract_parses(matrix[:, b], [lengths[b]] * matrix.shape[0], kbest=False, inc=inc)
        else:
            span, tree, lprob = extract_parse(matrix[b], lengths[b], inc=inc)
        trees.append(tree)
        spans.append(span)
        lprobs.append(lprob)
    return spans, trees, lprobs


def get_random_tree(length, SHIFT=0, REDUCE=1):
    tree = [SHIFT, SHIFT]
    stack = ['', '']
    num_shift = 2
    while len(tree) < 2 * length - 1:
        if len(stack) < 2:
            tree.append(SHIFT)
            stack.append('')
            num_shift += 1
        elif num_shift >= length:
            tree.append(REDUCE)
            stack.pop()
        else:
            if np.random.random_sample() < 0.5:
                tree.append(SHIFT)
                stack.append('')
                num_shift += 1
            else:
                tree.append(REDUCE)
                stack.pop()
    return tree

def get_left_branching_tree(length, SHIFT=0, REDUCE=1):
    tree = [SHIFT, SHIFT]
    stack = ['', '']
    num_shift = 2
    while len(tree) < 2 * length - 1:
        if len(stack) < 2:
            tree.append(SHIFT)
            stack.append('')
            num_shift += 1
        elif num_shift >= length:
            tree.append(REDUCE)
            stack.pop()
        else:
            tree.append(REDUCE)
            stack.pop()
    return tree

def get_right_branching_tree(length, SHIFT=0, REDUCE=1):
    tree = [SHIFT, SHIFT]
    stack = ['', '']
    num_shift = 2
    while len(tree) < 2 * length - 1:
        if len(stack) < 2:
            tree.append(SHIFT)
            stack.append('')
            num_shift += 1
        elif num_shift >= length:
            tree.append(REDUCE)
            stack.pop()
        else:
            tree.append(SHIFT)
            stack.append('')
            num_shift += 1
    return tree

def get_spans(actions, SHIFT = 0, REDUCE = 1):
  sent = list(range((len(actions)+1) // 2))
  spans = []
  pointer = 0
  stack = []
  if actions == [SHIFT, SHIFT]:
      return [(0, 0)]
  for action in actions:
    if action == SHIFT:
      word = sent[pointer]
      stack.append(word)
      pointer += 1
    elif action == REDUCE:
      right = stack.pop()
      left = stack.pop()
      if isinstance(left, int):
        left = (left, None)
      if isinstance(right, int):
        right = (None, right)
      new_span = (left[0], right[1])
      spans.append(new_span)
      stack.append(new_span)
  return spans