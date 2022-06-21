#ifndef _BENCH_HPP_
#define _BENCH_HPP_

#include<numeric>
#include<vector>

namespace benchmark {
  template<typename T>
  struct stats_t
  {
    T total;
    T mean;
    T min;
    T max;
  };

  template<typename T>
  inline stats_t<T> calculateStatistics(std::vector<T>& dataset)
  {
    stats_t<T> s;
    std::sort(dataset.begin(), dataset.end());
    s.total = std::accumulate(dataset.begin(), dataset.end(),T(0));
    s.mean = s.total / static_cast<T>(dataset.size());
    s.min = dataset.front();
    s.max = dataset.back();

    return s;
  }
}
#endif
