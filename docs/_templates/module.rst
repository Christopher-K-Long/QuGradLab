..
  module.rst

{{ name | escape | underline }}

.. automodule:: {{ fullname }}

    {% block attributes %}
        {% if attributes %}
            .. rubric:: Module Attributes

            .. autosummary::
                :toctree:
                {% for item in attributes %}
                    ~{{ fullname }}.{{ item }}
                {%- endfor %}
        {% endif %}
    {% endblock %}

    {% block functions %}
        {% if functions %}
            .. rubric:: {{ _('Functions') }}

            .. autosummary::
                :toctree:
                :nosignatures:
                :template: base.rst
                {% for item in functions %}
                    ~{{ fullname }}.{{ item }}
                {%- endfor %}
        {% endif %}
    {% endblock %}

    {% block classes %}
        {% if classes %}
            .. rubric:: {{ _('Classes') }}

            .. autosummary::
                :toctree:
                :nosignatures:
                :template: class.rst
                {% for item in classes %}
                    ~{{ fullname }}.{{ item }}
                {%- endfor %}
        {% endif %}
    {% endblock %}

    {% block exceptions %}
        {% if exceptions %}
            .. rubric:: {{ _('Exceptions') }}

            .. autosummary::
                :toctree:
                :nosignatures:
                :template: base.rst
                {% for item in exceptions %}
                    ~{{ fullname }}.{{ item }}
                {%- endfor %}
        {% endif %}
    {% endblock %}

{% block modules %}
    {% if modules %}
        .. rubric:: Modules

        .. autosummary::
            :toctree:
            :template: module.rst
            :recursive:
            {% for item in modules %}
                ~{{ fullname }}.{{ item }}
            {%- endfor %}
    {% endif %}
{% endblock %}