def package_outputs(stock_id, dates, outputs, version=0):
    ret = []
    for date, output in zip(dates, outputs):
        ret.append(('0', date, str(stock_id), "{:.4f}".format(output[0]), str(version)))
    return ret
