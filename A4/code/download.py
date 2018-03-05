import urllib2
for i in ['gru', 'lstm']:
	for j in ['50', '100', '200']:
		for k in ['checkpoint', 'model.data-00000-of-00001', 'model.index', 'model.meta']:
			fn = 'logs_' + i + '_hl' + j + '/' + k
			print 'Downloading file: %s' % fn
			if k == 'checkpoint':
				m = 'w'
				url = 'https://raw.githubusercontent.com/TheGrayFrost/Deep-Learning/master/A4/code/' + fn
			else:
				m = 'wb'
				url = 'https://github.com/TheGrayFrost/Deep-Learning/raw/master/A4/code/' + fn
			fp = urllib2.urlopen(url)
			with open(fn, m) as output:
				output.write(fp.read())
			print 'File downloaded'