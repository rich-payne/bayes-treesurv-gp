function out = get_grad_predictive0(f,ab, ns, C)
    kq = length(ns);
    dBdf = diag(exp(C * f)) * C;
    out = ones(1, kq) * (diag(ns) * C - dBdf .* ab);
    out = out';
end