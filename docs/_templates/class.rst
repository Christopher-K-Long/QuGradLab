..
  class.rst

{{ name | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
    :members:
    :show-inheritance:
    :inherited-members:
    :private-members: _pre_processing, _envolope_processing

    {% block attributes %}
        {% if attributes %}
            .. rubric:: {{ _('Attributes') }}

            .. autosummary::
                {% for item in attributes %}
                    ~{{ name }}.{{ item }}
                {%- endfor %}
        {% endif %}
    {% endblock %}

    {% block methods %}
        {% if methods %}
            .. rubric:: {{ _('Methods') }}

            .. autosummary::
                :nosignatures:
                {% for item in all_methods %}
                    {%- if not item.startswith('_') or item in ['__init__',
                                                                '_pre_processing',
                                                                '_envolope_processing'] %}
                    ~{{ name }}.{{ item }}
                    {%- endif -%}
                {%- endfor %}
        {% endif %}
    {% endblock %}