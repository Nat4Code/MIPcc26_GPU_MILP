/*
 * baseline_solve.c
 *
 * Plain Gurobi C API solve baseline with final JSON output for benchmarking.
 *
 * Usage:
 *   ./baseline model.mps [time_limit_seconds] [threads]
 *
 * Examples:
 *   ./baseline tests/instance_01.original.mps
 *   ./baseline tests/instance_01.original.mps 320
 *   ./baseline tests/instance_01.original.mps 320 16
 *
 * Notes:
 * - JSON is printed LAST so benchmark scripts can parse the final line.
 * - OutputFlag=1 keeps normal Gurobi progress visible for incumbent trace parsing.
 */

#include "gurobi_c.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

/* ---------- timing helpers ---------- */

static double wall_now_sec(void) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double)tv.tv_sec + 1e-6 * (double)tv.tv_usec;
}

/* ---------- JSON helpers ---------- */

static void json_print_escaped_string(const char *s) {
  putchar('"');
  if (s) {
    for (const unsigned char *p = (const unsigned char*)s; *p; ++p) {
      unsigned char c = *p;
      switch (c) {
        case '\\': fputs("\\\\", stdout); break;
        case '"':  fputs("\\\"", stdout); break;
        case '\n': fputs("\\n", stdout); break;
        case '\r': fputs("\\r", stdout); break;
        case '\t': fputs("\\t", stdout); break;
        default:
          if (c < 0x20) {
            printf("\\u%04x", (unsigned)c);
          } else {
            putchar((int)c);
          }
      }
    }
  }
  putchar('"');
}

static const char* status_name(int st) {
  switch (st) {
    case GRB_LOADED: return "LOADED";
    case GRB_OPTIMAL: return "OPTIMAL";
    case GRB_INFEASIBLE: return "INFEASIBLE";
    case GRB_INF_OR_UNBD: return "INF_OR_UNBD";
    case GRB_UNBOUNDED: return "UNBOUNDED";
    case GRB_CUTOFF: return "CUTOFF";
    case GRB_ITERATION_LIMIT: return "ITERATION_LIMIT";
    case GRB_NODE_LIMIT: return "NODE_LIMIT";
    case GRB_TIME_LIMIT: return "TIME_LIMIT";
    case GRB_SOLUTION_LIMIT: return "SOLUTION_LIMIT";
    case GRB_INTERRUPTED: return "INTERRUPTED";
    case GRB_NUMERIC: return "NUMERIC";
    case GRB_SUBOPTIMAL: return "SUBOPTIMAL";
    case GRB_INPROGRESS: return "INPROGRESS";
    default: return "STATUS_UNKNOWN";
  }
}

static void set_error_msg(char *buf, size_t n, const char *prefix, int error, GRBenv *env) {
  const char *grb_msg = NULL;
  if (env != NULL) {
    grb_msg = GRBgeterrormsg(env);
  }

  if (grb_msg != NULL && grb_msg[0] != '\0') {
    snprintf(buf, n, "%s failed, error=%d, gurobi_msg=%s", prefix, error, grb_msg);
  } else {
    snprintf(buf, n, "%s failed, error=%d", prefix, error);
  }
}

int main(int argc, char *argv[])
{
  int error = 0;
  GRBenv   *env   = NULL;
  GRBmodel *model = NULL;

  const char *model_file = NULL;
  double time_limit = -1.0;  /* < 0 means: no limit set */
  int threads = 16;          /* default benchmark setting */
  const char *log_file = "baseline_solve.log";

  /* benchmark JSON fields */
  int ok = 0;
  int status = -1;
  int terminated_early = 0;
  double best_obj = 0.0;
  int have_obj = 0;
  double best_bound = 0.0;
  int have_bound = 0;
  double mip_gap = 0.0;
  int have_gap = 0;
  double wall_runtime = 0.0;
  double grb_runtime = -1.0;
  char err_buf[4096];
  err_buf[0] = '\0';

  double t0 = wall_now_sec();

  if (argc < 2) {
    snprintf(err_buf, sizeof(err_buf), "Usage: baseline model.mps [time_limit_seconds] [threads]");
    goto QUIT;
  }

  model_file = argv[1];

  if (argc >= 3) {
    time_limit = atof(argv[2]);
  }

  if (argc >= 4) {
    threads = atoi(argv[3]);
    if (threads <= 0) threads = 16;
  }

  /*
   * Use emptyenv/startenv instead of GRBloadenv so we can:
   *   1. set params before the env starts,
   *   2. get better diagnostics if startup fails,
   *   3. avoid obscure GRBloadenv behavior with some WLS/container setups.
   */
  error = GRBemptyenv(&env);
  if (error) {
    set_error_msg(err_buf, sizeof(err_buf), "GRBemptyenv", error, env);
    goto QUIT;
  }

  error = GRBsetstrparam(env, GRB_STR_PAR_LOGFILE, log_file);
  if (error) {
    set_error_msg(err_buf, sizeof(err_buf), "GRBsetstrparam(LogFile)", error, env);
    goto QUIT;
  }

  error = GRBsetintparam(env, GRB_INT_PAR_OUTPUTFLAG, 1);
  if (error) {
    set_error_msg(err_buf, sizeof(err_buf), "GRBsetintparam(OutputFlag)", error, env);
    goto QUIT;
  }

  error = GRBsetintparam(env, GRB_INT_PAR_THREADS, threads);
  if (error) {
    set_error_msg(err_buf, sizeof(err_buf), "GRBsetintparam(Threads)", error, env);
    goto QUIT;
  }

  if (time_limit > 0.0) {
    error = GRBsetdblparam(env, GRB_DBL_PAR_TIMELIMIT, time_limit);
    if (error) {
      set_error_msg(err_buf, sizeof(err_buf), "GRBsetdblparam(TimeLimit)", error, env);
      goto QUIT;
    }
  }

  error = GRBstartenv(env);
  if (error) {
    set_error_msg(err_buf, sizeof(err_buf), "GRBstartenv", error, env);
    goto QUIT;
  }

  /* Read model from file. */
  error = GRBreadmodel(env, model_file, &model);
  if (error) {
    set_error_msg(err_buf, sizeof(err_buf), "GRBreadmodel", error, env);
    goto QUIT;
  }

  /* Optimize model. */
  error = GRBoptimize(model);
  if (error) {
    set_error_msg(err_buf, sizeof(err_buf), "GRBoptimize", error, env);
    goto QUIT;
  }

  /* Get solve status. */
  error = GRBgetintattr(model, GRB_INT_ATTR_STATUS, &status);
  if (error) {
    set_error_msg(err_buf, sizeof(err_buf), "GRBgetintattr(Status)", error, env);
    goto QUIT;
  }

  if (status == GRB_TIME_LIMIT ||
      status == GRB_NODE_LIMIT ||
      status == GRB_ITERATION_LIMIT ||
      status == GRB_INTERRUPTED) {
    terminated_early = 1;
  }

  /* Get incumbent objective when available. */
  if (status == GRB_OPTIMAL || status == GRB_TIME_LIMIT || status == GRB_SUBOPTIMAL ||
      status == GRB_NODE_LIMIT || status == GRB_ITERATION_LIMIT || status == GRB_INTERRUPTED) {
    double objval = 0.0;
    int e2 = GRBgetdblattr(model, GRB_DBL_ATTR_OBJVAL, &objval);
    if (!e2) {
      best_obj = objval;
      have_obj = 1;
    }
  }

  /* Optional bound and gap fields. */
  {
    double b = 0.0;
    int e2 = GRBgetdblattr(model, GRB_DBL_ATTR_OBJBOUND, &b);
    if (!e2) {
      best_bound = b;
      have_bound = 1;
    }
  }

  {
    double g = 0.0;
    int e2 = GRBgetdblattr(model, GRB_DBL_ATTR_MIPGAP, &g);
    if (!e2) {
      mip_gap = g;
      have_gap = 1;
    }
  }

  /* Gurobi-reported runtime. */
  {
    double rt = 0.0;
    int e2 = GRBgetdblattr(model, GRB_DBL_ATTR_RUNTIME, &rt);
    if (!e2) grb_runtime = rt;
  }

  ok = 1;

QUIT:
  wall_runtime = wall_now_sec() - t0;

  /* Always print JSON LAST. */
  {
    printf("{");
    printf("\"tool\":\"baseline\",");
    printf("\"ok\":%s,", ok ? "true" : "false");

    printf("\"mps\":");
    json_print_escaped_string(model_file ? model_file : "");
    printf(",");

    printf("\"time_limit_sec\":%.6f,", time_limit);
    printf("\"threads\":%d,", threads);
    printf("\"terminated_early\":%s,", terminated_early ? "true" : "false");

    printf("\"status_code\":%d,", status);
    printf("\"status\":");
    json_print_escaped_string(status_name(status));
    printf(",");

    printf("\"orbit_count\":null,");

    if (have_obj) {
      printf("\"best_obj\":%.17g,", best_obj);
      printf("\"objective\":%.17g,", best_obj);
    } else {
      printf("\"best_obj\":null,");
      printf("\"objective\":null,");
    }

    if (have_bound) {
      printf("\"best_bound\":%.17g,", best_bound);
    } else {
      printf("\"best_bound\":null,");
    }

    if (have_gap) {
      printf("\"mip_gap\":%.17g,", mip_gap);
    } else {
      printf("\"mip_gap\":null,");
    }

    printf("\"wall_runtime_sec\":%.9f,", wall_runtime);
    if (grb_runtime >= 0.0) {
      printf("\"grb_runtime_sec\":%.9f", grb_runtime);
    } else {
      printf("\"grb_runtime_sec\":null");
    }

    if (!ok) {
      printf(",\"error_msg\":");
      json_print_escaped_string(err_buf[0] ? err_buf : "unknown error");
    }

    printf("}\n");
    fflush(stdout);
  }

  if (model) GRBfreemodel(model);
  if (env)   GRBfreeenv(env);

  return ok ? 0 : 1;
}